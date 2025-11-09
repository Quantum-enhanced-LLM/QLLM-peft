from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments
import torch
import wandb
import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
print('load_dotenv', load_dotenv(Path.cwd() / '.env'))

class QPeftLogCallback(TrainerCallback):
    def __init__(self, qpeft_arch : str):
        super().__init__()
        self.grad_norms = {}
        self.model = None
        self.full_grads = {}
        self.full_forwards = {}
        if qpeft_arch == 'ABC':
            self.target_param_keywords = ["lora_qpeft_Q", 
                                          "lora_qpeft_MPO_A",
                                          "lora_qpeft_MPO_B",
                                          "lora_qpeft_MLP_A",
                                          "lora_qpeft_MLP_B",
                                          "lora_qpeft_CW",
                                          "lora_qpeft_QW"]
            
            self.target_param_value = ["lora_qpeft_CW.default",
                                       "lora_qpeft_QW.default"]
            
        elif qpeft_arch == 'BC':
            self.target_param_keywords = ["lora_qpeft_Q", 
                                          "lora_qpeft_MLP_A",
                                          "lora_qpeft_MLP_B",
                                          "lora_qpeft_CW",
                                          "lora_qpeft_QW"] 
            
            self.target_param_value = ["lora_qpeft_CW.default",
                                       "lora_qpeft_QW.default"]

        elif qpeft_arch == 'C':
            self.target_param_keywords = ["lora_qpeft_Q", 
                                          "lora_qpeft_MLP_A",
                                          "lora_qpeft_MLP_B",
                                          "lora_qpeft_QW"] 
            
            self.target_param_value = ["lora_qpeft_QW.default"]
                 
        else:
            raise ValueError("QPeftLogCallback only applies to ABC/BC/C archs.")


    def _create_hook(self, param_name: str):
        def hook(grad):
            if grad is not None:
                if f"train/grad_norm_{param_name}" in self.grad_norms:
                    self.grad_norms[f"train/grad_norm_{param_name}"] += torch.linalg.norm(grad).item()
                else:
                    self.grad_norms[f"train/grad_norm_{param_name}"] = torch.linalg.norm(grad).item()

                if f"train/grad_mean_{param_name}" in self.grad_norms:
                    self.grad_norms[f"train/grad_mean_{param_name}"] += torch.mean(grad).item()
                else:
                    self.grad_norms[f"train/grad_mean_{param_name}"] = torch.mean(grad).item()

        return hook
    
    def _create_back_hook_detail(self, full_name: str):
        def hook(grad):
            if grad is not None:
                self.full_grads[full_name] = grad.cpu().numpy()

        return hook                

    def _create_forward_hook(self, layer_name: str):
        raise ValueError("We should not call this.")
        
    def _create_forward_hook_detail(self, layer_name: str):
        def hook(module, input_tensor, output_tensor):
            if isinstance(output_tensor, torch.Tensor):
                target_tensor = output_tensor
            elif isinstance(output_tensor, tuple) and len(output_tensor) > 0 and isinstance(output_tensor[0], torch.Tensor):
                # If output is a tuple, let's take the first tensor.
                target_tensor = output_tensor[0]                
                if len(output_tensor) > 1:
                    print(f"Warning (LayerOutputLogCallback): Layer '{layer_name}' output is a tuple. Logging stats for the first element. Full output structure: {[type(o) for o in output_tensor]}")
            else:
                raise ValueError(f"Cannot identify this. output_tensor = {output_tensor}")
            
            if target_tensor is not None:
                # Detach and move to CPU to avoid holding onto GPU memory and computation graph
                data_to_store = data_to_store = target_tensor.detach().clone().float().cpu().numpy()

                # Store the collected data, keyed by layer name and then by stat                     
                self.full_forwards[layer_name] = data_to_store

        return hook
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        self.model = model
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                for keyword in self.target_param_keywords:
                    if keyword in name:
                        param.register_hook(self._create_hook(keyword))
                        if int(os.getenv('OUTPUT_GRAD_DETAILS', 0)):
                            param.register_hook(self._create_back_hook_detail(name))
                        print(f"  - Backward Hook registered for {keyword}: {name}")
                        break
        
        if int(os.getenv('OUTPUT_FORWARD_DETAILS', 0)):
            for name, module in model.named_modules():
                for keyword in self.target_param_value:
                    if name.endswith(keyword):
                        module.register_forward_hook(self._create_forward_hook_detail(name))
                        print(f"  - Forward Hook registered for {keyword}: {name}")                        
                        break


    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("qpeft grads", self.grad_norms)

        if self.grad_norms:
            if wandb.run: # Check if a wandb run is active
                wandb.log({**self.grad_norms.copy(), "train/global_step": state.global_step})
        
        # for full grads, save in the local directory using the global_step as the filename
        if self.full_grads:
            working_directory = args.output_dir
            filename = f"grads_{state.global_step}.pkl"
            filepath = os.path.join(working_directory, filename)
            with open(filepath, "wb") as f:
                pickle.dump(self.full_grads, f)

        if self.full_forwards:
            working_directory = args.output_dir
            filename = f"forwards_{state.global_step}.pkl"
            filepath = os.path.join(working_directory, filename)
            with open(filepath, "wb") as f:
                pickle.dump(self.full_forwards, f)

        self.grad_norms.clear()
        self.full_grads.clear()
        self.full_forwards.clear()


class QPeftLogForwardCallback(TrainerCallback):
    def __init__(self, qpeft_arch : str):
        super().__init__()
        self.collected_outputs = {}
        self.model = None
        self.step = 0
        if qpeft_arch == 'ABC':
            self.target_param_value = ["lora_qpeft_CW.default",
                                       "lora_qpeft_QW.default"]
            
        elif qpeft_arch == 'BC':
            self.target_param_value = ["lora_qpeft_CW.default",
                                       "lora_qpeft_QW.default"]
            
        elif qpeft_arch == 'C':
            self.target_param_value = ["lora_qpeft_QW.default"]
        
        else:
            raise ValueError("QPeftLogForwardCallback only applies to ABC/BC/C archs.")

    def _create_forward_hook(self, layer_name: str):
        def hook(module, input_tensor, output_tensor):
            if isinstance(output_tensor, torch.Tensor):
                target_tensor = output_tensor
            elif isinstance(output_tensor, tuple) and len(output_tensor) > 0 and isinstance(output_tensor[0], torch.Tensor):
                # If output is a tuple, let's take the first tensor.
                target_tensor = output_tensor[0]                
                if len(output_tensor) > 1:
                    print(f"Warning (LayerOutputLogCallback): Layer '{layer_name}' output is a tuple. Logging stats for the first element. Full output structure: {[type(o) for o in output_tensor]}")
            else:
                raise ValueError(f"Cannot identify this. output_tensor = {output_tensor}")
            
            if target_tensor is not None:
                data_to_store = target_tensor.detach().clone()
                self.collected_outputs[layer_name] = data_to_store.float().cpu().numpy()
                
        return hook
        
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        self.model = model
        for name, module in model.named_modules():
            for keyword in self.target_param_value:
                if name.endswith(keyword):
                    module.register_forward_hook(self._create_forward_hook(name))
                    print(f"  - Forward Hook registered for {keyword}: {name}")
                    
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(self.collected_outputs)
        print(self.step)
        if self.collected_outputs:
            working_directory = args.output_dir
            filename = f"forward_{self.step}.pkl"
            filepath = os.path.join(working_directory, filename)
            with open(filepath, "wb") as f:
                pickle.dump(self.collected_outputs, f)

        self.step += 1
        self.collected_outputs.clear()
        