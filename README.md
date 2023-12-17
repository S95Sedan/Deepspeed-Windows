DeepSpeed Version 11.2 with CUDA 12.1 - Installation Instructions:

1. Download the 11.2 release of [DeepSpeed](https://github.com/microsoft/DeepSpeed/releases/tag/v0.11.2) extract it to a folder. 
2. Install Visual C++ build tools, such as [VS2019 C++ x64/x86](https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#vs2019-download) build tools.
3. Download and install the [Nvidia Cuda Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
4. Edit your Windows environment variables to ensure that CUDA_HOME and CUDA_PATH are set to your Nvidia Cuda Toolkit path. (The folder above the bin folder that nvcc.exe is installed in). Examples are:<br>
```set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1```<br>
```set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1```<br>

5. OPTIONAL If you do not have an python environment already created, you can install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html), then at a command prompt, create and activate your environment with:<br>
```conda create -n pythonenv python=3.11```<br>
```activate pythonenv```<br>

6. Launch the Command Prompt cmd with Administrator privilege as it requires admin to allow creating symlink folders.
7. Install PyTorch, 2.1.0 with CUDA 12.1 into your Python 3.11 environment e.g:<br>
```activate pythonenv``` (activate your python environment)<br>
```conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia```

8. In your python environment check that your CUDA_HOME and CUDA_PATH are still pointing to the correct location.<br>
```set``` (to list and check the windows environment variables. Refer to step 4 if not)

9. Navigate to your deepspeed folder in the Command Prompt:<br>
```cd c:\deepspeed``` (wherever you extracted it to)

10. Modify the following files:<br>

 deepspeed-0.11.2/build_win.bat - at the top of the file, add:<br>
 ```set DS_BUILD_EVOFORMER_ATTN=0```

deepspeed-0.11.2/csrc/quantization/pt_binding.cpp - lines 244-250 - change to:
```
    std::vector<int64_t> sz_vector(input_vals.sizes().begin(), input_vals.sizes().end());
    sz_vector[sz_vector.size() - 1] = sz_vector.back() / devices_per_node;  // num of GPU per nodes
    at::IntArrayRef sz(sz_vector);
    auto output = torch::empty(sz, output_options);

    const int elems_per_in_tensor = at::numel(input_vals) / devices_per_node;
    const int elems_per_in_group = elems_per_in_tensor / (in_groups / devices_per_node);
    const int elems_per_out_group = elems_per_in_tensor / out_groups;
```

deepspeed-0.11.2/csrc/transformer/inference/csrc/pt_binding.cpp
lines 541-542 - change to:
```
									 {static_cast<unsigned>(hidden_dim * InferenceContext::Instance().GetMaxTokenLength()),
									  static_cast<unsigned>(k * InferenceContext::Instance().GetMaxTokenLength()),
```

lines 550-551 - change to:
```
						 {static_cast<unsigned>(hidden_dim * InferenceContext::Instance().GetMaxTokenLength()),
						  static_cast<unsigned>(k * InferenceContext::Instance().GetMaxTokenLength()),
```
line 1581 - change to:
```
		at::from_blob(intermediate_ptr, {input.size(0), input.size(1), static_cast<int64_t>(mlp_1_out_neurons)}, options);
```

deepspeed-0.11.2/deepspeed/env_report.py
line 10 - add:
```
import psutil
```
line 83 - 100 - change to:
```
def get_shm_size():
    try:
        temp_dir = os.getenv('TEMP') or os.getenv('TMP') or os.path.join(os.path.expanduser('~'), 'tmp')
        shm_stats = psutil.disk_usage(temp_dir)
        shm_size = shm_stats.total
        shm_hbytes = human_readable_size(shm_size)
        warn = []
        if shm_size < 512 * 1024**2:
            warn.append(
                f" {YELLOW} [WARNING] Shared memory size might be too small, consider increasing it. {END}"
            )
            # Add additional warnings specific to your use case if needed.
        return shm_hbytes, warn
    except Exception as e:
        return "UNKNOWN", [f"Error getting shared memory size: {e}"]
```

11. While still in your command line with python environment enabled run:<br>
```build_win.bat```

12. Once you are done building there should be a .whl file is present in:<br>
```deepspeed-0.11.2/dist/```

13. Copy that file to the root of your Oobabooga folder and run:<br>
```cmd_windows.bat```<br>
```pip install deepspeed-YOURFILENAME.whl``` (Or whichever name your .whl has you just created)

14. To check if its working correctly you can type the following:<br>
```bash```
```ds_report```
