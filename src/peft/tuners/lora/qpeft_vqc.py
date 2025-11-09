import torchquantum as tq
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
import pyqpanda3.core as pq
from pyqpanda3.core import NoiseModel, depolarizing_error, GateType    
from pyqpanda3.transpilation import Transpiler,generate_topology
import os
import itertools
import sqlite3
import hashlib
import json
from pyqpanda3.qcloud import QCloudService, QCloudOptions
from dotenv import load_dotenv
from pathlib import Path
print('load_dotenv', load_dotenv(Path.cwd() / '.env'))

def batched(iterable, n):
    """
    Equivalent to itertools.batched in Python 3.12+.
    Groups elements from an iterable into batches of size n.
    The last batch may be smaller than n.
    """
    if n < 1:
        raise ValueError('n must be at least one')
    
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            pass # Iterator exhausted

        if not chunk:
            return # No more elements, stop
        
        yield list(chunk) # Yield the current batch

class VQC(tq.QuantumModule):
    def __init__(self, 
                 n_wires: int = 8,
                 n_qlayers: int = 1,
    ):
        super().__init__()
        self.n_wires = n_wires 
        self.n_qlayers = n_qlayers
        self.backend = os.getenv('BACKEND', 'torchquantum')
        # verify backend
        if self.backend not in ['torchquantum', 'vqnet', 'vqnet_virtual', 'vqnet_noisy']:
            raise ValueError("Cannot identify qpeft_backend: ", self.backend)

        if int(os.getenv('USE_CACHE', 0)) and self.backend != 'torchquantum':
            self.db_path = f'{self.backend}_{self.n_wires}_cache.db'        
            self._setup_cache_db()

        self.shots = int(os.getenv('SHOTS', 1000))
        if self.backend in ['vqnet', 'vqnet_noisy'] and self.shots <= 0:
            raise ValueError(f"Must have a valid shot number with BACKEND={self.backend}")

        if self.backend == 'vqnet':            
            option = QCloudOptions()
            option.set_amend(True)
            option.set_mapping(True)
            option.set_optimization(True)
            self.qcloud_option = option
        
        if self.backend == 'vqnet_virtual':
            self.m_machine = pq.CPUQVM()
        
        if self.backend == 'vqnet_noisy':
            self.topology = generate_topology(20, "square")
            self.transpiler = Transpiler()
            self.basic_gates = ['U3', 'CZ']
            def _transpile(prog):
                return self.transpiler.transpile(prog, self.topology, {}, 2, self.basic_gates)

            self.transpile = _transpile
            self.m_machine = pq.CPUQVM()
            self.noise_model = NoiseModel()
            self.noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01), GateType.U3)
            self.noise_model.add_all_qubit_quantum_error(depolarizing_error(0.1), GateType.CZ)
            self.noise_model.add_all_qubit_read_out_error([[0.95,0.05],[0.1,0.9]])
        
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)
        
        self.params_ry1_dct = tq.QuantumModuleDict()
        self.params_ry2_dct = tq.QuantumModuleDict()
        self.params_crx1_dct = tq.QuantumModuleDict()
        self.params_crx2_dct = tq.QuantumModuleDict()
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_ry1_dct[str(i + k*self.n_wires)] = tq.RY(has_params=True, trainable=True)
                self.params_crx1_dct[str(i + k*self.n_wires)] = tq.CRZ(has_params=True, trainable=True)
                self.params_ry2_dct[str(i + k*self.n_wires)] = tq.RY(has_params=True, trainable=True)
                self.params_crx2_dct[str(i + k*self.n_wires)] = tq.CRZ(has_params=True, trainable=True)
 
        self.measure = tq.MeasureMultipleTimes([{'wires': range(self.n_wires), 'observables': ['z'] * self.n_wires}])
        self.dev = tq.QuantumDevice(n_wires=self.n_wires)

    def run_torchquantum(self, x):
        q_device = self.dev
        q_device.reset_states(x.shape[0])
        
        for k in range(self.n_qlayers):
            self.encoder(q_device, x)
                
            for i in range(self.n_wires - 1, -1, -1):
                self.params_crx1_dct[str(i + k*self.n_wires)](q_device, wires=[i, (i + 1) % self.n_wires])

            for i in range(self.n_wires):
                self.params_ry1_dct[str(i + k*self.n_wires)](q_device, wires=i)            
            
            for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                self.params_crx2_dct[str(i + k*self.n_wires)](q_device, wires=[i, (i - 1) % self.n_wires])

            for i in range(self.n_wires):
                self.params_ry2_dct[str(i + k*self.n_wires)](q_device, wires=i)

        output_tq = self.measure(q_device)
        return output_tq

    def make_vqnet_circuit(self, input):
        circuit = pq.QCircuit()
        input = input.detach().to(torch.float).cpu().numpy()
        
        # print('vqnet')
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                circuit<<pq.RY(i, input[i])
                
            for i in range(0, self.n_wires, 2):
                param = self.params_crx1_dct[str(i + k*self.n_wires)].params.detach().to(torch.float)
                circuit<<pq.RZ((i + 1) % self.n_wires, param.cpu().numpy()).control(i)
                
            for i in range(1, self.n_wires, 2):
                param = self.params_crx1_dct[str(i + k*self.n_wires)].params.detach().to(torch.float)
                circuit<<pq.RZ((i + 1) % self.n_wires, param.cpu().numpy()).control(i)
                
            for i in range(self.n_wires):
                param = self.params_ry1_dct[str(i + k*self.n_wires)].params.detach().to(torch.float)
                circuit<<pq.RY(i, param.cpu().numpy())
                
            for i in range(0, self.n_wires, 2):
                param = self.params_crx2_dct[str(i + k*self.n_wires)].params.detach().to(torch.float)
                circuit<<pq.RZ((i - 1) % self.n_wires, param.cpu().numpy()).control(i)
                
            for i in range(1, self.n_wires, 2):
                param = self.params_crx2_dct[str(i + k*self.n_wires)].params.detach().to(torch.float)
                circuit<<pq.RZ((i - 1) % self.n_wires, param.cpu().numpy()).control(i)
        
            for i in range(self.n_wires):
                param = self.params_ry2_dct[str(i + k*self.n_wires)].params.detach().to(torch.float)
                circuit<<pq.RY(i, param.cpu().numpy())
                
        prog = pq.QProg()
        prog<<circuit

        if self.shots > 0:
            prog << pq.measure(list(range(self.n_wires)), list(range(self.n_wires)))
        return prog

    def _setup_cache_db(self):
        print(f"Cache DB path: {os.path.abspath(self.db_path)}")
        self.db_conn = sqlite3.connect(self.db_path)
        self.db_cursor = self.db_conn.cursor()
        
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS circuit_cache (
                circuit_hash TEXT PRIMARY KEY,
                result_json TEXT NOT NULL
            )
        ''')
        self.db_conn.commit()

    def _get_circuit_hash(self, prog):
        from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog
        oir = prog.originir(precision=4) 
        return hashlib.sha256(oir.encode('utf-8')).hexdigest(), convert_originir_string_to_qprog(oir)
    
    def run_vqnet_virtual(self, x):        
        # reset state
        # x should be [batchsize, inputsize]
        bsz = x.shape[0]
        output_qprogs = [self.make_vqnet_circuit(x[i,:]) for i in range(x.shape[0])]
        results = []

        if int(os.getenv('USE_CACHE', 0)):
            for prog in output_qprogs:
                cache_key, prog = self._get_circuit_hash(prog)
                self.db_cursor.execute("SELECT result_json FROM circuit_cache WHERE circuit_hash = ?", (cache_key,))
                cached_result = self.db_cursor.fetchone()
                if cached_result:
                    expvals = np.array(json.loads(cached_result[0]))
                    print('Cached result = ', expvals)
                    results.append(expvals)
                else:                
                    self.m_machine.run(prog, self.shots)
                    counts = self.m_machine.result().get_prob_dict()
                    expvals = np.zeros(self.n_wires)
                    for key in counts:
                        val = counts[key]
                        for i, digit in enumerate(key):
                            if digit == '0':
                                expvals[self.n_wires - 1 - i] += val
                            else:
                                expvals[self.n_wires - 1 - i] -= val

                    result_json = json.dumps(expvals.tolist())
                    self.db_cursor.execute(
                        "INSERT INTO circuit_cache (circuit_hash, result_json) VALUES (?, ?)",
                        (cache_key, result_json)
                    )
                    self.db_conn.commit()
                    results.append(expvals)
        else:            
            for prog in output_qprogs:
                self.m_machine.run(prog, self.shots)
                counts = self.m_machine.result().get_prob_dict()
                expvals = np.zeros(self.n_wires)
                for key in counts:
                    val = counts[key]
                    for i, digit in enumerate(key):
                        if digit == '0':
                            expvals[self.n_wires - 1 - i] += val
                        else:
                            expvals[self.n_wires - 1 - i] -= val               
                results.append(expvals)
        return np.array(results)
    
    def run_vqnet_noisy(self, x):        
        # reset state
        # x should be [batchsize, inputsize]
        bsz = x.shape[0]
        output_qprogs = [self.make_vqnet_circuit(x[i,:]) for i in range(x.shape[0])]
        results = []
        if int(os.getenv('USE_CACHE', 0)):
            for prog in output_qprogs:
                cache_key, prog = self._get_circuit_hash(prog)
                self.db_cursor.execute("SELECT result_json FROM circuit_cache WHERE circuit_hash = ?", (cache_key,))
                cached_result = self.db_cursor.fetchone()
                if cached_result:
                    expvals = np.array(json.loads(cached_result[0]))
                    print('Cached result = ', expvals)
                    results.append(expvals)
                else:                
                    prog = self.transpile(prog)
                    self.m_machine.run(prog, self.shots, self.noise_model)
                    counts = self.m_machine.result().get_prob_dict()
                    expvals = np.zeros(self.n_wires)
                    for key in counts:
                        val = counts[key]
                        for i, digit in enumerate(key):
                            if digit == '0':
                                expvals[self.n_wires - 1 - i] += val
                            else:
                                expvals[self.n_wires - 1 - i] -= val

                    result_json = json.dumps(expvals.tolist())
                    self.db_cursor.execute(
                        "INSERT INTO circuit_cache (circuit_hash, result_json) VALUES (?, ?)",
                        (cache_key, result_json)
                    )
                    self.db_conn.commit()
                    results.append(expvals)
        else:
            for prog in output_qprogs:
                prog = self.transpile(prog)
                self.m_machine.run(prog, self.shots, self.noise_model)
                counts = self.m_machine.result().get_prob_dict()
                expvals = np.zeros(self.n_wires)
                for key in counts:
                    val = counts[key]
                    for i, digit in enumerate(key):
                        if digit == '0':
                            expvals[self.n_wires - 1 - i] += val
                        else:
                            expvals[self.n_wires - 1 - i] -= val

                results.append(expvals)
                
        return np.array(results)
        
    def run_vqnet(self, x):
        # reset state
        # x should be [batchsize, inputsize]
        output_qprogs = [self.make_vqnet_circuit(x[i,:]) for i in range(x.shape[0])]

        progs_to_run = []
        hashes_to_run = []
        cached_results = {} 
        original_hashes_order = [] 
        
        print("Step 1: Checking cache...")
        for prog in output_qprogs:
            prog_hash, prog = self._get_circuit_hash(prog)
            original_hashes_order.append(prog_hash)

            self.db_cursor.execute("SELECT result_json FROM circuit_cache WHERE circuit_hash = ?", (prog_hash,))
            cached_result = self.db_cursor.fetchone()

            if cached_result:
                if prog_hash not in cached_results:
                     cached_results[prog_hash] = np.array(json.loads(cached_result[0]))
            else:
                if prog_hash not in hashes_to_run:
                    progs_to_run.append(prog)
                    hashes_to_run.append(prog_hash)
        
        print(f"Cache check complete. Hit: {len(output_qprogs) - len(progs_to_run)}, Miss: {len(progs_to_run)}")
        
        if progs_to_run:
            print(f"Step 2: Submitting {len(progs_to_run)} circuits to cloud...")
            apikey = os.getenv('API_KEY')
            real_chip_name = os.getenv('REAL_CHIP_NAME', 'WK_C102_400')
            circuit_batch_count = int(os.getenv('CIRCUIT_BATCH_COUNT', 200))

            if apikey is None:
                raise ValueError("API_KEY is not set.")

            service = QCloudService(apikey)
            m_machine = service.backend(real_chip_name)
            
            newly_computed_counts = []
            for prog_batch in batched(progs_to_run, circuit_batch_count):
                print(f"  - Submitting {len(prog_batch)} circuit in a batch...")
                job = m_machine.run(prog_batch, self.shots, self.qcloud_option)
                newly_computed_counts.extend(job.result().get_probs_list())

            print("Step 3: Processing new results and updating cache...")
            for i, counts in enumerate(newly_computed_counts):
                prog_hash = hashes_to_run[i]
                
                expvals = np.zeros(self.n_wires)
                for key, val in counts.items():
                    for j, digit in enumerate(key):
                        if digit == '0':
                            expvals[self.n_wires - 1 - j] += val
                        else:
                            expvals[self.n_wires - 1 - j] -= val
                
                cached_results[prog_hash] = expvals                
                result_json = json.dumps(expvals.tolist())
                self.db_cursor.execute(
                    "INSERT OR IGNORE INTO circuit_cache (circuit_hash, result_json) VALUES (?, ?)",
                    (prog_hash, result_json)
                )
            self.db_conn.commit()
            print("Cache update complete.")

        print("Step 4: Rearranging results to match original input order...")
        final_results = [cached_results[h] for h in original_hashes_order]
        return np.array(final_results)

    @tq.static_support 
    def forward(self, x: torch.Tensor):
        """
        1. To convert tq QuantumModule to qiskit or run in the static model,
        we need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and 
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        output_tq = self.run_torchquantum(x)

        if self.backend == 'torchquantum':
            return (output_tq)
        elif self.backend == 'vqnet_virtual':
            print('vqnet_virtual')
            output_vqnet = self.run_vqnet_virtual(x)
            # check the shape and results
            if output_tq.shape != output_vqnet.shape:
                print("   output_tq.shape =", output_tq.shape)
                print("output_vqnet.shape =", output_vqnet.shape)
                raise ValueError("Cannot match shape")
            if self.shots == 0 and (not np.allclose(output_tq.detach().to(torch.float).cpu().numpy(), output_vqnet, atol=0.1)):
                print('   output_tq = ', output_tq)
                print('output_vqnet = ', output_vqnet)
                raise ValueError("Cannot match results.")
            print(output_tq.shape, output_vqnet.shape)

            return (torch.from_numpy(output_vqnet).to('cuda:0'))
        elif self.backend == 'vqnet_noisy':
            print('vqnet_noisy')
            output_vqnet = self.run_vqnet_noisy(x)
            # check the shape and results
            if output_tq.shape != output_vqnet.shape:
                print("   output_tq.shape =", output_tq.shape)
                print("output_vqnet.shape =", output_vqnet.shape)
                raise ValueError("Cannot match shape")

            return (torch.from_numpy(output_vqnet).to('cuda:0'))
        elif self.backend == 'vqnet':
            print('vqnet')
            while True:
                output_vqnet = self.run_vqnet(x)
                if output_tq.shape != output_vqnet.shape:
                    print("   output_tq.shape =", output_tq.shape)
                    print("output_vqnet.shape =", output_vqnet.shape)
                    raise ValueError("Cannot match shape")
                                
                return (torch.from_numpy(output_vqnet).to('cuda:0'))
        else:
            raise NotImplementedError("Not implemented yet.")
        
    def close(self):
        if self.db_conn:
            self.db_conn.close()

    def __del__(self):
        self.close()

class QLP(nn.Module):
    def __init__(self, 
                 n_qubits: int = 8, 
                 n_qlayers=1, 
                 backend: Optional[str] = None,
                 shots: Optional[int] = 1000):
        super(QLP, self).__init__()
        self.n_qubits = n_qubits
        # Using a classical feed-forward layer
        self.vqc = VQC(n_wires=n_qubits, n_qlayers=n_qlayers)
        
    def forward(self, input_features):
        new_input_features = input_features.clone()
        if new_input_features.ndim == 3:
            bsz,len,fz = input_features.size()
            input_features = input_features.view(-1, fz)
        else:
            bsz,fz = input_features.size()
 
        q_in = input_features
        quantum_out = self.vqc(q_in)
        if new_input_features.ndim == 3:
            output = quantum_out.view(bsz,len,fz)
        else:
             output = quantum_out.view(bsz,fz)
        return output


class TTOLayer(nn.Module):
    def __init__(self, 
                 inp_modes, 
                 out_modes, 
                 mat_ranks,
                 cores_initializer=nn.init.kaiming_normal_, 
                 cores_regularizer=None, 
                 biases_initializer=torch.zeros, 
                 biases_regularizer=None, 
                 trainable=True, 
                 cpu_variables=False, 
                 scope=None):
        super(TTOLayer, self).__init__()
        
        self.inp_modes = inp_modes
        self.out_modes = out_modes
        self.mat_ranks = mat_ranks
        self.dim = len(inp_modes)
        self.cpu_variables = cpu_variables
        
        self.mat_cores = nn.ParameterList()
        for i in range(self.dim):
            shape = (out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i])
            core = torch.empty(shape)
            cores_initializer(core)
            self.mat_cores.append(nn.Parameter(core, requires_grad=trainable))
    
    def forward(self, inp):
        batch_size, len, embed_size = inp.shape        
        out = inp.view(-1, np.prod(self.inp_modes))
        out = out.t()
        
        for i in range(self.dim):            
            out = out.reshape(self.mat_ranks[i] * self.inp_modes[i], -1)
            out = torch.matmul(self.mat_cores[i], out.to(self.mat_cores[i].dtype))
            out = out.reshape(self.out_modes[i], -1)
            out = out.t()        
      
        out = out.reshape(batch_size, len, np.prod(self.out_modes))
        return out