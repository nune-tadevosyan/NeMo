import copy
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.parts.utils.run_ipl_utils import *
from nemo.core.config import hydra_runner
from nemo.utils import logging


def get_command_for_inference(
    inference_config: str, inference_config_dir: Union[str, Path], p_cache: float, checkpoint: str
) -> Tuple[str, List[str], List[str]]:
    """
    Generates the command string for running speech inference with transcribe_speech_parallel.

    Args:
        inference_config (str): Path to the base inference configuration file.
        inference_config_dir (Union[str, Path]): Directory to store temporary modified configurations.
        p_cache (float): Proportion of the dataset to be cached for pseudo-labeling.
        checkpoint (str): Path to the model checkpoint to use for inference.

    Returns:
        Tuple[str, List[str], List[str]]:
            - The command string to execute inference for all specified manifests.
            - List of output directories corresponding to each manifest.
            - List of completed full pass transcribed manifest paths, if any.
    """
    manifests, tarr_audio_files = separate_multiple_transcriptions(inference_config)
    num_gpus = torch.cuda.device_count()
    output_dirs = []
    cmd = ""
    print(f"manifests {manifests}")
    print(f"tarr_audio_files {tarr_audio_files}")
    for i in range(len(manifests)):
        output_dir = os.path.dirname(manifests[i])
        output_dirs.append(output_dir)

        base_cfg = OmegaConf.load(inference_config)
        temp_config_dir = Path(str(inference_config_dir) + "/temp_configs").absolute()
        os.makedirs(temp_config_dir, exist_ok=True)
        modified_cfg = copy.deepcopy(base_cfg)

        # Check if we need to run inference on the whole set or update part of it
        full_pass_done = glob.glob(os.path.join(output_dir, 'transcribed_manifest*'))
        if full_pass_done:
            number_of_files = count_files_for_pseudo_labeling(manifests[i], bool(tarr_audio_files))
            limit_predict_batches = int((number_of_files * p_cache) / (modified_cfg.predict_ds.batch_size * num_gpus))
            OmegaConf.update(modified_cfg, "trainer.limit_predict_batches", limit_predict_batches)

        # Replace OmegaConf updates with simple assignments
        OmegaConf.update(modified_cfg, "output_path", output_dir)
        OmegaConf.update(modified_cfg, "predict_ds.manifest_filepath", manifests[i])
        if tarr_audio_files:
            OmegaConf.update(modified_cfg, "predict_ds.tarred_audio_filepaths", tarr_audio_files[i])
        OmegaConf.update(modified_cfg, "model", checkpoint)

        temp_config_file = os.path.join(temp_config_dir, f"modified_config_{i}.yaml")
        OmegaConf.save(modified_cfg, temp_config_file)
        cmd += f"python examples/asr/transcribe_speech_parallel.py --config-path {temp_config_dir} --config-name modified_config_{i}.yaml && "

    # Remove trailing '&&' from the final command string
    cmd = cmd.rstrip(" &&")

    print(f"Inference command: {cmd}")
    return cmd, output_dirs, full_pass_done


def get_execution_script(cluster_script_path: str, config_name: str, config_path: str, updated_manifest_filepaths=None, updated_tarred_filepaths=None) -> str:
    """
    Constructs a command string to execute a training with the specified configuration.

    Args:
        cluster_script_path (str): Path to the cluster script to be executed.
        config_name (str): Name of the configuration file or object to be passed as a parameter.
        config_path (str): Path to the directory where the configuration resides.

    Returns:
        str: A formatted command string ready for execution.
    """
    # Create the command to run the script
    cmd = """
        cd {cluster_script_dir} && \
        python {cluster_script_path} --config-path {config_path} --config-name "{config_name}" """
    format_dict = dict(
        cluster_script_dir=os.path.dirname(cluster_script_path),
        cluster_script_path=os.path.basename(cluster_script_path),
        config_path=config_path,
        config_name=config_name,
    )
    cmd = cmd.format(**format_dict)
    if updated_manifest_filepaths:
        cmd += f" model.train_ds.manifest_filepath={updated_manifest_filepaths}"
    if updated_tarred_filepaths:
        cmd += f" model.train_ds.tarred_audio_filepaths={updated_tarred_filepaths}"
    print(f"Training command: {cmd}")
    return cmd


def find_checkpoint_dir(base_path):
    """
    Find the 'checkpoints' folder in the directory structure.
    Parameters:
        base_path (str): The base directory path to search from.
    """
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "checkpoints":
                return os.path.join(root, dir_name), root
    return None, None


def run_command(cmd: str, shell: bool = True, log_file: str = None) -> bool:
    """
    Safely run a shell command using subprocess and stream output in real-time.
    
    Args:
        cmd (str): Command to execute
        shell (bool): Whether to use shell for command execution
        log_file (str): Optional path to save logs to file
        
    Returns:
        bool: True if command executed successfully, False otherwise
    """
    try:
        # Create log file if specified
        log_handle = None
        if log_file:
            log_handle = open(log_file, 'a')
            log_handle.write(f"\n{'='*80}\n")
            log_handle.write(f"Command: {cmd}\n")
            log_handle.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write(f"{'='*80}\n\n")

        # Start the process
        process = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                if log_handle:
                    log_handle.write(output)
                    log_handle.flush()

        # Get any remaining stderr
        stderr = process.stderr.read()
        if stderr:
            print(f"Error output: {stderr}", file=sys.stderr)
            if log_handle:
                log_handle.write(f"Error output: {stderr}\n")
                log_handle.flush()

        # Get return code
        return_code = process.poll()
        
        if log_handle:
            log_handle.write(f"\nProcess completed with return code: {return_code}\n")
            log_handle.close()

        return return_code == 0

    except Exception as e:
        print(f"Command failed with error: {e}")
        if log_handle:
            log_handle.write(f"Command failed with error: {e}\n")
            log_handle.close()
        return False


@hydra_runner(config_path='./', config_name='run_ipl')
def main(run_config):
    script_config = run_config.script_config
    script_path = run_config.script_path
    inference_config = run_config.inference_config
    ipl_epochs = run_config.ipl_epochs
    inference_config_dir = os.path.dirname(Path(inference_config).absolute())
    script_config_path = os.path.dirname(Path(script_config).absolute())
    script_config_name = os.path.basename(Path(script_config).absolute())
    inference_config = os.path.join(inference_config_dir, inference_config)

    # Create logs directory
    logs_dir = os.path.join(script_config_path, "ipl_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"ipl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Load the config directly
    config = OmegaConf.load(script_config)
    config.exp_manager.resume_if_exists = True
    
    # Find initial checkpoint

    add_pl_datasets = True
    updated_manifest_filepaths=None
    updated_tarred_audio_filepaths=None
    for epoch in range(ipl_epochs):
        print(f"\nStarting IPL epoch {epoch + 1}/{ipl_epochs}")
        
        # First run training
        training_command = get_execution_script(script_path, script_config_name, script_config_path, updated_manifest_filepaths, updated_tarred_audio_filepaths)
        run_command(training_command, log_file=log_file)

        # Update checkpoint after training
        checkpoint_path, logs_dir = find_checkpoint_dir(
            os.path.join(config.exp_manager.exp_dir, config.exp_manager.name)
        )
        checkpoint = os.path.join(checkpoint_path, config.exp_manager.name + ".nemo")
        
        # Then run inference
        cmd, output_dirs, full_pass_done = get_command_for_inference(
            inference_config, inference_config_dir, 0.5, checkpoint
        )
        if not run_command(cmd, log_file=log_file):
            print("Inference failed, stopping IPL process")
            break

        # Create manifests based on whether it's first pass or not
        if not full_pass_done:
            if config.model.train_ds.is_tarred:
                all_manifest_filepaths = create_transcribed_shard_manifests(output_dirs)
            else:
                all_manifest_filepaths = create_transcribed_manifests(output_dirs)
        else:
            if config.model.train_ds.is_tarred:
                all_manifest_filepaths = write_sampled_shard_transcriptions(output_dirs)
            else:
                all_manifest_filepaths = write_sampled_transcriptions(output_dirs)

        # Update training sets if needed
        if add_pl_datasets:
            base_cfg = OmegaConf.load(inference_config)
            updated_manifest_filepaths, updated_tarred_audio_filepaths  = update_training_sets(
                config, all_manifest_filepaths, base_cfg.predict_ds.get("tarred_audio_filepaths", None)
            )
            add_pl_datasets = False

        # Save updated config for next iteration
        config_filepath = os.path.join(script_config_path, "update_script_config.yaml")
        OmegaConf.save(config, config_filepath)
        
        print(f"Completed IPL epoch {epoch + 1}/{ipl_epochs}")


if __name__ == '__main__':
    main()
