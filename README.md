# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

## Task 4.5
### Task 4.5.1: Training Log for [run_sentiment.py](project/run_sentiment.py)
**[sentiment.txt](sentiment.txt)**
```
missing pre-trained embedding for 55 unknown words
Epoch 1, loss 31.530143843790725, train accuracy: 43.11%
Validation accuracy: 48.00%
Best Valid accuracy: 48.00%
Epoch 2, loss 31.31748839204681, train accuracy: 51.11%
Validation accuracy: 46.00%
Best Valid accuracy: 48.00%
Epoch 3, loss 31.085679735750823, train accuracy: 51.33%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 4, loss 30.88330839984807, train accuracy: 55.33%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 5, loss 30.603136490554903, train accuracy: 57.33%
Validation accuracy: 62.00%
Best Valid accuracy: 63.00%
Epoch 6, loss 30.51201963460498, train accuracy: 57.33%
Validation accuracy: 60.00%
Best Valid accuracy: 63.00%
Epoch 7, loss 30.06840501635106, train accuracy: 61.11%
Validation accuracy: 61.00%
Best Valid accuracy: 63.00%
Epoch 8, loss 29.82441325677897, train accuracy: 62.89%
Validation accuracy: 61.00%
Best Valid accuracy: 63.00%
Epoch 9, loss 29.242202324654166, train accuracy: 63.78%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 10, loss 28.77020930387169, train accuracy: 68.00%
Validation accuracy: 67.00%
Best Valid accuracy: 67.00%
Epoch 11, loss 28.39542177037869, train accuracy: 69.56%
Validation accuracy: 63.00%
Best Valid accuracy: 67.00%
Epoch 12, loss 28.125957886332067, train accuracy: 68.67%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 13, loss 27.30767366650305, train accuracy: 72.22%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 14, loss 26.96516643926035, train accuracy: 70.44%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 15, loss 25.83890820724738, train accuracy: 74.89%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 16, loss 25.81587745627731, train accuracy: 71.56%
Validation accuracy: 71.00%
Best Valid accuracy: 73.00%
Epoch 17, loss 25.14403830372215, train accuracy: 73.33%
Validation accuracy: 72.00%
Best Valid accuracy: 73.00%
Epoch 18, loss 23.981040686055, train accuracy: 76.67%
Validation accuracy: 71.00%
Best Valid accuracy: 73.00%
Epoch 19, loss 24.193111424519426, train accuracy: 75.33%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 20, loss 23.106791953271244, train accuracy: 74.22%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 21, loss 22.464484939840304, train accuracy: 76.22%
Validation accuracy: 69.00%
Best Valid accuracy: 73.00%
Epoch 22, loss 21.898267116190393, train accuracy: 76.89%
Validation accuracy: 70.00%
Best Valid accuracy: 73.00%
Epoch 23, loss 21.87781098457725, train accuracy: 78.89%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 24, loss 20.532729626044755, train accuracy: 77.33%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 25, loss 20.27587162853242, train accuracy: 78.22%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 26, loss 19.096519129433272, train accuracy: 81.78%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 27, loss 19.49467364851724, train accuracy: 80.89%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 28, loss 18.56772562866756, train accuracy: 79.33%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 29, loss 18.077951210684596, train accuracy: 81.78%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 30, loss 17.502723050640423, train accuracy: 82.67%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 31, loss 17.440126144663736, train accuracy: 81.33%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 32, loss 16.49804628901665, train accuracy: 84.00%
Validation accuracy: 70.00%
Best Valid accuracy: 75.00%
Epoch 33, loss 16.500501371904473, train accuracy: 81.56%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 34, loss 16.22000472927484, train accuracy: 80.89%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 35, loss 15.856132292762675, train accuracy: 83.11%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 36, loss 14.614782320527528, train accuracy: 84.89%
Validation accuracy: 72.00%
Best Valid accuracy: 75.00%
Epoch 37, loss 14.271771204736341, train accuracy: 83.56%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
Epoch 38, loss 14.827086895973814, train accuracy: 83.56%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 39, loss 14.71265654262626, train accuracy: 83.78%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 40, loss 13.52653760765778, train accuracy: 87.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 41, loss 14.345102556169675, train accuracy: 83.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 42, loss 13.479492375150706, train accuracy: 86.44%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 43, loss 13.125233196298083, train accuracy: 85.33%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 44, loss 12.844580355759057, train accuracy: 88.67%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 45, loss 12.723122681577589, train accuracy: 86.00%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 46, loss 12.593623191511805, train accuracy: 85.11%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 47, loss 11.971514493466982, train accuracy: 86.67%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 48, loss 12.370384730391377, train accuracy: 87.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 49, loss 12.126904200014982, train accuracy: 85.33%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 50, loss 11.33817892353693, train accuracy: 86.89%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 51, loss 12.194414688701762, train accuracy: 84.89%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 52, loss 11.627074602250117, train accuracy: 86.44%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 53, loss 11.889725436637457, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 54, loss 12.188651271997808, train accuracy: 85.78%
Validation accuracy: 75.00%
Best Valid accuracy: 76.00%
Epoch 55, loss 12.306188203669588, train accuracy: 85.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 56, loss 10.811704106057707, train accuracy: 85.78%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 57, loss 12.225228513700957, train accuracy: 86.00%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 58, loss 10.125236197657301, train accuracy: 88.44%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 59, loss 10.339268982362988, train accuracy: 88.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 60, loss 10.70207294013589, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 61, loss 10.693753561722179, train accuracy: 88.22%
Validation accuracy: 69.00%
Best Valid accuracy: 76.00%
Epoch 62, loss 10.561539936126913, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 63, loss 10.554722560499785, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 64, loss 11.200677200809611, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 65, loss 10.4692669259019, train accuracy: 88.00%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 66, loss 10.263437461541537, train accuracy: 88.00%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 67, loss 9.270196778124417, train accuracy: 88.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 68, loss 10.413420424709939, train accuracy: 85.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 69, loss 8.651245145354608, train accuracy: 88.22%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 70, loss 9.500811529003553, train accuracy: 88.22%
Validation accuracy: 70.00%
Best Valid accuracy: 76.00%
Epoch 71, loss 9.884065138058606, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 72, loss 8.308206018877632, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 73, loss 10.804119174305397, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 74, loss 9.999089752512537, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 75, loss 10.327737950045638, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 76, loss 8.932335936138065, train accuracy: 89.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 77, loss 8.328859572330677, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 78, loss 8.994186722518686, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 79, loss 8.835157367360278, train accuracy: 86.44%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 80, loss 8.997366487144365, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 81, loss 9.860491241704889, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 82, loss 9.058463336580278, train accuracy: 86.44%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 83, loss 9.244748838565371, train accuracy: 88.22%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 84, loss 9.305387413731845, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 85, loss 9.600584825359098, train accuracy: 86.22%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 86, loss 9.302183401277402, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 87, loss 7.256643004492623, train accuracy: 90.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 88, loss 8.954396829492424, train accuracy: 87.11%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 89, loss 9.423284430583982, train accuracy: 86.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 90, loss 10.039150193425266, train accuracy: 86.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 91, loss 8.816295161219102, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 92, loss 9.4763488800971, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 93, loss 10.13152484297565, train accuracy: 84.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 94, loss 8.888868770908928, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 95, loss 8.607563799147439, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 96, loss 8.587075703845775, train accuracy: 87.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 97, loss 8.893230251782539, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 98, loss 8.461329095660728, train accuracy: 89.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 99, loss 9.15184273029474, train accuracy: 85.56%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 100, loss 8.936009965833177, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 101, loss 9.149820246292236, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 102, loss 8.306424015986629, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 103, loss 8.82903723768835, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 104, loss 8.49058762532259, train accuracy: 88.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 105, loss 8.954865458853376, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 106, loss 8.646097156899792, train accuracy: 86.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 107, loss 8.189611874497924, train accuracy: 88.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 108, loss 8.92662313102725, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 109, loss 8.60074822546027, train accuracy: 85.78%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 110, loss 7.785707329086468, train accuracy: 88.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 111, loss 10.101066700466683, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 112, loss 8.446096923043381, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 113, loss 9.03806025031243, train accuracy: 87.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 114, loss 9.037225405734787, train accuracy: 87.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 115, loss 8.822876576895545, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 116, loss 8.801854311833049, train accuracy: 86.44%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 117, loss 7.973085263162464, train accuracy: 87.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 118, loss 9.22065797546349, train accuracy: 86.44%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 119, loss 9.977956458766316, train accuracy: 82.89%
Validation accuracy: 74.00%
Best Valid accuracy: 76.00%
Epoch 120, loss 8.235340247770814, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 121, loss 8.899535799249149, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 122, loss 8.935342628776215, train accuracy: 84.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 123, loss 8.895919970968126, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 124, loss 8.417567662955147, train accuracy: 86.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 125, loss 8.575194263374842, train accuracy: 85.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 126, loss 7.979995578845211, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 127, loss 9.401230833627638, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 128, loss 8.674137550027272, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 129, loss 8.324405498999186, train accuracy: 87.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 130, loss 8.755101745877399, train accuracy: 86.44%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 131, loss 7.8033486983812255, train accuracy: 89.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 132, loss 9.196047885159823, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 133, loss 7.5463832540944, train accuracy: 88.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 134, loss 8.953499673529555, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 135, loss 8.35755283032613, train accuracy: 87.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 136, loss 8.179037508532629, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 137, loss 8.449169853058345, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 138, loss 8.084235888231126, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 139, loss 8.715067949149999, train accuracy: 85.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 140, loss 8.523291435216631, train accuracy: 85.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 141, loss 8.594551434010134, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 142, loss 8.273198633770454, train accuracy: 86.89%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 143, loss 8.456997025442178, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 144, loss 8.7973978312594, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 145, loss 8.951609329433294, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 146, loss 8.77687813876299, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 147, loss 8.604895593128838, train accuracy: 84.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 148, loss 9.751515066634465, train accuracy: 84.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 149, loss 8.611458897385944, train accuracy: 89.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 150, loss 8.487047555633628, train accuracy: 85.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 151, loss 7.412530551684397, train accuracy: 89.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 152, loss 8.797077992227768, train accuracy: 85.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 153, loss 8.014511000557299, train accuracy: 87.33%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 154, loss 8.246969550838443, train accuracy: 85.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 155, loss 9.08049442874918, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 156, loss 9.147740547565428, train accuracy: 84.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 157, loss 9.493439585674707, train accuracy: 85.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 158, loss 8.04702832174641, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 159, loss 6.9900551287612815, train accuracy: 88.22%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 160, loss 9.718003206039183, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 161, loss 8.537963143658482, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 162, loss 7.643725263433497, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 163, loss 7.694721444302709, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 164, loss 8.329764562044778, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 165, loss 8.423328604362885, train accuracy: 88.67%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 166, loss 7.890402034670112, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 167, loss 9.55673833159369, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 168, loss 7.996947303583246, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 169, loss 9.13851222537433, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 170, loss 7.987878177162648, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 171, loss 8.444234528845456, train accuracy: 83.56%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 172, loss 8.039750718098162, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 173, loss 7.872139650728179, train accuracy: 89.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 174, loss 8.624697037220148, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 175, loss 8.83318881426394, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 176, loss 8.21533635876677, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 177, loss 9.20572805151872, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 178, loss 8.292718918414185, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 179, loss 7.264977773449099, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 180, loss 7.378327017770702, train accuracy: 89.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 181, loss 7.424842629242034, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 182, loss 8.023090914346021, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 183, loss 7.458135762442722, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 184, loss 8.5814770536171, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 185, loss 7.097695939314046, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 186, loss 8.610061235389733, train accuracy: 87.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 187, loss 8.920106197989545, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 188, loss 7.9025457435855975, train accuracy: 89.11%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 189, loss 8.113028950130056, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 190, loss 8.383827125406167, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 191, loss 9.58441559772836, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 192, loss 7.209779890937004, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 193, loss 7.482086773256354, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 194, loss 6.92348373817886, train accuracy: 90.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 195, loss 7.069877242300021, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 196, loss 8.43975516519343, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 197, loss 7.284753467912572, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 198, loss 8.894857702599937, train accuracy: 85.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 199, loss 9.374131734150671, train accuracy: 84.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 200, loss 8.344491129100835, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 201, loss 8.678979147103451, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 202, loss 7.613966360992498, train accuracy: 88.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 203, loss 8.889157504661197, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 204, loss 7.958749507176413, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 205, loss 9.235382107416042, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 206, loss 8.45592727277589, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 207, loss 8.274328790674787, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 208, loss 7.534470829412489, train accuracy: 89.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 209, loss 8.055318487392277, train accuracy: 87.33%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 210, loss 7.833134188538268, train accuracy: 88.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 211, loss 8.271980203040394, train accuracy: 87.78%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 212, loss 8.386836172716642, train accuracy: 88.00%
Validation accuracy: 73.00%
Best Valid accuracy: 76.00%
Epoch 213, loss 7.778610611293843, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 214, loss 8.63642533069836, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 215, loss 9.067587216184938, train accuracy: 83.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 216, loss 7.847735623073412, train accuracy: 86.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 217, loss 7.6806487166115005, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 218, loss 10.224917215757815, train accuracy: 83.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 219, loss 8.106705171693674, train accuracy: 86.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 220, loss 6.797735027481153, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 221, loss 7.736058400772668, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 222, loss 7.416277545422035, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 223, loss 7.963013175586866, train accuracy: 85.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 224, loss 8.549703477785306, train accuracy: 83.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 225, loss 7.068556095071119, train accuracy: 89.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 226, loss 7.624691231566508, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 227, loss 8.58461308317338, train accuracy: 86.22%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 228, loss 9.09711110979272, train accuracy: 85.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 229, loss 9.269201573827976, train accuracy: 83.56%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 230, loss 7.491695797446918, train accuracy: 88.44%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 231, loss 8.163049260440369, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 232, loss 7.196655105387503, train accuracy: 87.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 233, loss 8.408411166822386, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 234, loss 8.475315658386332, train accuracy: 88.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 235, loss 6.848964583848407, train accuracy: 89.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 236, loss 8.201520984257474, train accuracy: 86.44%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 237, loss 8.988329516842269, train accuracy: 87.56%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 238, loss 6.931023304793351, train accuracy: 88.67%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 239, loss 8.541336612129864, train accuracy: 87.33%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 240, loss 8.691769388041457, train accuracy: 85.11%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 241, loss 8.48225095320257, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 242, loss 7.77948661644024, train accuracy: 85.11%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 243, loss 9.36330004730303, train accuracy: 84.67%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 244, loss 6.942986014308628, train accuracy: 88.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 245, loss 7.9870675893640595, train accuracy: 86.89%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 246, loss 7.546804975194019, train accuracy: 87.78%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 247, loss 7.669342477172414, train accuracy: 87.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 248, loss 8.2760373729386, train accuracy: 88.00%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
Epoch 249, loss 8.13667444504638, train accuracy: 86.00%
Validation accuracy: 72.00%
Best Valid accuracy: 76.00%
Epoch 250, loss 9.63719990064028, train accuracy: 83.78%
Validation accuracy: 71.00%
Best Valid accuracy: 76.00%
```

### Task 4.5.2: Training Log for [run_mnist_multiclass.py](project/run_mnist_multiclass.py)
**[mnist.txt](mnist.txt)**
```
Epoch 1 loss 2.296558666714976 valid acc 3/16
Epoch 1 loss 11.497657756007758 valid acc 2/16
Epoch 1 loss 11.482881061313252 valid acc 2/16
Epoch 1 loss 11.524982252674778 valid acc 3/16
Epoch 1 loss 11.525120713431926 valid acc 3/16
Epoch 1 loss 11.431838430424746 valid acc 2/16
Epoch 1 loss 11.445062417590709 valid acc 4/16
Epoch 1 loss 11.519239509181006 valid acc 4/16
Epoch 1 loss 11.468338450688117 valid acc 4/16
Epoch 1 loss 11.386674350249422 valid acc 5/16
Epoch 1 loss 11.409875845821432 valid acc 5/16
Epoch 1 loss 11.366624866000265 valid acc 6/16
Epoch 1 loss 11.32536018180674 valid acc 6/16
Epoch 1 loss 10.928277796376143 valid acc 6/16
Epoch 1 loss 10.618173648135674 valid acc 9/16
Epoch 1 loss 9.38397333448748 valid acc 9/16
Epoch 1 loss 10.978679540841169 valid acc 4/16
Epoch 1 loss 9.860766910989573 valid acc 7/16
Epoch 1 loss 8.625436408728861 valid acc 10/16
Epoch 1 loss 7.406712226589004 valid acc 12/16
Epoch 1 loss 6.662011408810915 valid acc 12/16
Epoch 1 loss 5.767426357241797 valid acc 11/16
Epoch 1 loss 4.676826152849621 valid acc 11/16
Epoch 1 loss 5.1048458744332414 valid acc 12/16
Epoch 1 loss 5.1831556263642895 valid acc 7/16
Epoch 1 loss 5.286646019230055 valid acc 14/16
Epoch 1 loss 5.172616070365256 valid acc 14/16
Epoch 1 loss 2.951406453874915 valid acc 13/16
Epoch 1 loss 4.504656261675609 valid acc 14/16
Epoch 1 loss 2.8019001727896446 valid acc 12/16
Epoch 1 loss 3.6157390078210216 valid acc 11/16
Epoch 1 loss 3.9959785994327 valid acc 13/16
Epoch 1 loss 3.1386396731465336 valid acc 13/16
Epoch 1 loss 3.6399542269408385 valid acc 13/16
Epoch 1 loss 5.614443258213644 valid acc 13/16
Epoch 1 loss 3.7549388746696044 valid acc 12/16
Epoch 1 loss 3.448722273508504 valid acc 13/16
Epoch 1 loss 2.9966384365476277 valid acc 14/16
Epoch 1 loss 2.873752405475783 valid acc 13/16
Epoch 1 loss 2.952262540110835 valid acc 15/16
Epoch 1 loss 3.1567765600851923 valid acc 13/16
Epoch 1 loss 3.034034883506373 valid acc 14/16
Epoch 1 loss 2.026556534716681 valid acc 14/16
Epoch 1 loss 2.9841030948040803 valid acc 13/16
Epoch 1 loss 3.600287657441582 valid acc 16/16
Epoch 1 loss 2.291080328879107 valid acc 15/16
Epoch 1 loss 2.7587940821231967 valid acc 15/16
Epoch 1 loss 2.5825086440075498 valid acc 12/16
Epoch 1 loss 2.2341417351448234 valid acc 14/16
Epoch 1 loss 1.9945701098415713 valid acc 13/16
Epoch 1 loss 2.0391656940219716 valid acc 14/16
Epoch 1 loss 2.2089789321786437 valid acc 16/16
Epoch 1 loss 3.099171594843655 valid acc 14/16
Epoch 1 loss 1.95899608925112 valid acc 13/16
Epoch 1 loss 3.5275692539752135 valid acc 12/16
Epoch 1 loss 1.7314805232195334 valid acc 14/16
Epoch 1 loss 2.1731325584562335 valid acc 14/16
Epoch 1 loss 2.1371974580208217 valid acc 13/16
Epoch 1 loss 2.366407999261985 valid acc 15/16
Epoch 1 loss 2.05174950998605 valid acc 13/16
Epoch 1 loss 3.087025859108463 valid acc 14/16
Epoch 1 loss 2.9580327387671472 valid acc 14/16
Epoch 1 loss 2.127751741939838 valid acc 14/16
Epoch 2 loss 0.049355591530453824 valid acc 14/16
Epoch 2 loss 1.8317764084737906 valid acc 14/16
Epoch 2 loss 3.4121787023625476 valid acc 15/16
Epoch 2 loss 2.5807114857543283 valid acc 13/16
Epoch 2 loss 1.3716331692429908 valid acc 13/16
Epoch 2 loss 1.7765663743486102 valid acc 12/16
Epoch 2 loss 2.276522040899141 valid acc 13/16
Epoch 2 loss 2.819278624486719 valid acc 13/16
Epoch 2 loss 2.5641810236423836 valid acc 13/16
Epoch 2 loss 1.245693361729641 valid acc 14/16
Epoch 2 loss 2.1404415104663683 valid acc 14/16
Epoch 2 loss 3.143085584547515 valid acc 14/16
Epoch 2 loss 2.3540066756770246 valid acc 15/16
Epoch 2 loss 3.281200735949065 valid acc 13/16
Epoch 2 loss 2.244663624775983 valid acc 14/16
Epoch 2 loss 1.6816658392277264 valid acc 14/16
Epoch 2 loss 3.73763712964877 valid acc 13/16
Epoch 2 loss 2.5413476241421016 valid acc 15/16
Epoch 2 loss 2.0988874531320487 valid acc 14/16
Epoch 2 loss 1.6126942389607395 valid acc 13/16
Epoch 2 loss 2.149754768235516 valid acc 12/16
Epoch 2 loss 1.1978446705619237 valid acc 14/16
Epoch 2 loss 0.7442640599873056 valid acc 14/16
Epoch 2 loss 1.6058393991394775 valid acc 12/16
Epoch 2 loss 2.17296476212032 valid acc 14/16
Epoch 2 loss 1.2909965100641094 valid acc 14/16
Epoch 2 loss 1.1845966910929717 valid acc 15/16
Epoch 2 loss 1.5880107517418263 valid acc 14/16
Epoch 2 loss 1.3208452531060797 valid acc 15/16
Epoch 2 loss 0.6908740305449235 valid acc 14/16
Epoch 2 loss 1.1303389983241425 valid acc 14/16
Epoch 2 loss 1.7544662475294661 valid acc 12/16
Epoch 2 loss 0.7547093617317749 valid acc 13/16
Epoch 2 loss 1.5981188092749996 valid acc 14/16
Epoch 2 loss 3.9164599280715793 valid acc 14/16
Epoch 2 loss 2.031259127953013 valid acc 14/16
Epoch 2 loss 1.1813755563772832 valid acc 14/16
Epoch 2 loss 1.8689854284071796 valid acc 14/16
Epoch 2 loss 1.4940213924000136 valid acc 16/16
Epoch 2 loss 1.4483901848047005 valid acc 14/16
Epoch 2 loss 0.7260522673284973 valid acc 13/16
Epoch 2 loss 1.4099703359958342 valid acc 15/16
Epoch 2 loss 1.208290027510948 valid acc 15/16
Epoch 2 loss 0.9310391066280134 valid acc 15/16
Epoch 2 loss 2.367009071610502 valid acc 16/16
Epoch 2 loss 1.5619171225329955 valid acc 14/16
Epoch 2 loss 1.3176475941950425 valid acc 15/16
Epoch 2 loss 2.3585363809053463 valid acc 14/16
Epoch 2 loss 0.9945088874272358 valid acc 15/16
Epoch 2 loss 1.1346231229388046 valid acc 14/16
Epoch 2 loss 1.14498339287974 valid acc 14/16
Epoch 2 loss 1.100648047011182 valid acc 15/16
Epoch 2 loss 2.200982819620773 valid acc 14/16
Epoch 2 loss 1.0156520990511908 valid acc 14/16
Epoch 2 loss 1.492346498028541 valid acc 13/16
Epoch 2 loss 0.9400830346306643 valid acc 15/16
Epoch 2 loss 1.3917544789901282 valid acc 15/16
Epoch 2 loss 1.3051704491772542 valid acc 14/16
Epoch 2 loss 1.649851141939271 valid acc 14/16
Epoch 2 loss 0.9800225892399669 valid acc 14/16
Epoch 2 loss 1.1160481171682946 valid acc 14/16
Epoch 2 loss 0.9491037889570736 valid acc 15/16
Epoch 2 loss 0.9653288748236981 valid acc 15/16
Epoch 3 loss 0.02236156478044582 valid acc 15/16
Epoch 3 loss 1.3984749817409656 valid acc 14/16
Epoch 3 loss 2.2486584232897338 valid acc 16/16
Epoch 3 loss 1.035566508217674 valid acc 15/16
Epoch 3 loss 0.5898229130640058 valid acc 15/16
Epoch 3 loss 1.146321943412623 valid acc 16/16
Epoch 3 loss 1.6499551186658925 valid acc 14/16
Epoch 3 loss 1.4901054797359858 valid acc 14/16
Epoch 3 loss 1.1530836414309065 valid acc 15/16
Epoch 3 loss 0.8534270561644576 valid acc 15/16
Epoch 3 loss 0.9431925975210372 valid acc 15/16
Epoch 3 loss 2.263865844795646 valid acc 16/16
Epoch 3 loss 1.6443805204121422 valid acc 15/16
Epoch 3 loss 1.821430837026092 valid acc 16/16
Epoch 3 loss 1.9268454913975286 valid acc 16/16
Epoch 3 loss 0.688863374137451 valid acc 16/16
Epoch 3 loss 2.0935462681712584 valid acc 16/16
Epoch 3 loss 1.2233037844777173 valid acc 14/16
Epoch 3 loss 1.715090095228163 valid acc 15/16
Epoch 3 loss 1.0986050642555099 valid acc 15/16
Epoch 3 loss 1.4525762111096723 valid acc 14/16
Epoch 3 loss 0.8132489365514126 valid acc 15/16
Epoch 3 loss 0.5722831459170242 valid acc 16/16
Epoch 3 loss 1.6719799411055432 valid acc 13/16
Epoch 3 loss 2.1630685364660875 valid acc 15/16
Epoch 3 loss 1.134927250473525 valid acc 15/16
Epoch 3 loss 0.9875338821155414 valid acc 16/16
Epoch 3 loss 1.154686254843923 valid acc 16/16
Epoch 3 loss 0.8512971533188359 valid acc 15/16
Epoch 3 loss 0.41830946574318717 valid acc 16/16
Epoch 3 loss 1.4560171046165444 valid acc 15/16
Epoch 3 loss 1.6000939346311167 valid acc 15/16
Epoch 3 loss 0.5181262969427258 valid acc 16/16
Epoch 3 loss 1.0425275151542617 valid acc 16/16
Epoch 3 loss 1.6968521239898369 valid acc 14/16
Epoch 3 loss 1.3864214184560493 valid acc 15/16
Epoch 3 loss 0.8335889367611165 valid acc 14/16
Epoch 3 loss 0.9107254520799992 valid acc 14/16
Epoch 3 loss 1.0378129476092441 valid acc 15/16
Epoch 3 loss 0.8200962492267863 valid acc 15/16
Epoch 3 loss 0.5589134175070765 valid acc 13/16
Epoch 3 loss 1.055070600999894 valid acc 16/16
Epoch 3 loss 1.1821494569729023 valid acc 15/16
Epoch 3 loss 0.5541860946935665 valid acc 15/16
Epoch 3 loss 2.1588556929731775 valid acc 15/16
Epoch 3 loss 0.6724944662931076 valid acc 15/16
Epoch 3 loss 0.9657830791103182 valid acc 15/16
Epoch 3 loss 1.7506172785936625 valid acc 14/16
Epoch 3 loss 0.7151336232964034 valid acc 15/16
Epoch 3 loss 0.7166498556362798 valid acc 16/16
Epoch 3 loss 1.304377609248729 valid acc 16/16
Epoch 3 loss 0.9701232317019761 valid acc 16/16
Epoch 3 loss 1.0164379456691588 valid acc 16/16
Epoch 3 loss 0.8710784706860861 valid acc 16/16
Epoch 3 loss 1.3657248263595902 valid acc 16/16
Epoch 3 loss 0.3899350373347986 valid acc 16/16
Epoch 3 loss 0.5817771392621769 valid acc 16/16
Epoch 3 loss 1.1523026699562862 valid acc 15/16
Epoch 3 loss 1.590248520871845 valid acc 15/16
Epoch 3 loss 1.0429806126493473 valid acc 14/16
Epoch 3 loss 1.9635104603839242 valid acc 16/16
Epoch 3 loss 0.684228898985952 valid acc 15/16
Epoch 3 loss 0.9161990853852382 valid acc 15/16
Epoch 4 loss 0.018215884310671004 valid acc 15/16
Epoch 4 loss 0.6659987166107953 valid acc 14/16
Epoch 4 loss 1.8966509179572937 valid acc 16/16
Epoch 4 loss 0.9633086407436218 valid acc 15/16
Epoch 4 loss 0.5394411666019945 valid acc 14/16
Epoch 4 loss 0.798120038125326 valid acc 16/16
Epoch 4 loss 1.4683056196146804 valid acc 16/16
Epoch 4 loss 1.2392662545797992 valid acc 15/16
Epoch 4 loss 0.6868470707189974 valid acc 15/16
Epoch 4 loss 0.6210007869894586 valid acc 15/16
Epoch 4 loss 0.47126414955297924 valid acc 16/16
Epoch 4 loss 1.6558664855416716 valid acc 15/16
Epoch 4 loss 1.3718687494862922 valid acc 15/16
Epoch 4 loss 1.7000016576151347 valid acc 14/16
Epoch 4 loss 1.6522132831166132 valid acc 16/16
Epoch 4 loss 0.5645567907806175 valid acc 15/16
Epoch 4 loss 0.9095374323167464 valid acc 14/16
Epoch 4 loss 1.5151566363486855 valid acc 14/16
Epoch 4 loss 1.7891735173316115 valid acc 16/16
Epoch 4 loss 1.307317240125093 valid acc 16/16
Epoch 4 loss 0.6939134948664878 valid acc 16/16
Epoch 4 loss 0.7962814351214345 valid acc 16/16
Epoch 4 loss 0.34627636844493587 valid acc 16/16
Epoch 4 loss 0.7085475663119798 valid acc 16/16
Epoch 4 loss 1.0304797127553083 valid acc 14/16
Epoch 4 loss 1.093205117537436 valid acc 15/16
Epoch 4 loss 0.5905927502598607 valid acc 15/16
Epoch 4 loss 0.40597475023838325 valid acc 15/16
Epoch 4 loss 0.4959469287379456 valid acc 15/16
Epoch 4 loss 0.21605829024432235 valid acc 15/16
Epoch 4 loss 0.5279586137167923 valid acc 15/16
Epoch 4 loss 1.5657897273912498 valid acc 14/16
Epoch 4 loss 0.656373549712638 valid acc 15/16
Epoch 4 loss 1.633236245856942 valid acc 16/16
Epoch 4 loss 2.2549872015084356 valid acc 15/16
Epoch 4 loss 1.1534322197830027 valid acc 15/16
Epoch 4 loss 0.4484078179686593 valid acc 15/16
Epoch 4 loss 0.7857539578947452 valid acc 15/16
Epoch 4 loss 0.9302756233461462 valid acc 14/16
Epoch 4 loss 0.44415227527819473 valid acc 14/16
Epoch 4 loss 0.6429764631651136 valid acc 13/16
Epoch 4 loss 1.0088370386748808 valid acc 16/16
Epoch 4 loss 0.5095147662289681 valid acc 15/16
Epoch 4 loss 0.8485346110301535 valid acc 15/16
Epoch 4 loss 1.8930751725493549 valid acc 16/16
Epoch 4 loss 0.7658679285397952 valid acc 16/16
Epoch 4 loss 0.9014901626189492 valid acc 15/16
Epoch 4 loss 1.4469061073181575 valid acc 15/16
Epoch 4 loss 0.6744598219433426 valid acc 14/16
Epoch 4 loss 0.45922433935372164 valid acc 15/16
Epoch 4 loss 0.41121093973107653 valid acc 15/16
Epoch 4 loss 0.6238473410677883 valid acc 16/16
Epoch 4 loss 0.5794736957732983 valid acc 16/16
Epoch 4 loss 0.5334889965461167 valid acc 16/16
Epoch 4 loss 0.7375057219542569 valid acc 14/16
Epoch 4 loss 0.5691551602409619 valid acc 15/16
Epoch 4 loss 0.8468654487784247 valid acc 15/16
Epoch 4 loss 0.4092679007202189 valid acc 15/16
Epoch 4 loss 1.4209908114973935 valid acc 15/16
Epoch 4 loss 1.0296127208203378 valid acc 14/16
Epoch 4 loss 0.8472078505909908 valid acc 14/16
Epoch 4 loss 0.5126963914384672 valid acc 16/16
Epoch 4 loss 0.9273572999094617 valid acc 15/16
Epoch 5 loss 0.017571205687118785 valid acc 15/16
Epoch 5 loss 0.8961216165282424 valid acc 14/16
Epoch 5 loss 1.2362680674698798 valid acc 16/16
Epoch 5 loss 0.454872367013197 valid acc 15/16
Epoch 5 loss 0.4038541514135829 valid acc 14/16
Epoch 5 loss 0.5231241580037476 valid acc 14/16
Epoch 5 loss 1.1543911630675536 valid acc 16/16
Epoch 5 loss 1.1133291057036194 valid acc 16/16
Epoch 5 loss 0.5920196674691511 valid acc 15/16
Epoch 5 loss 0.32293382679142035 valid acc 15/16
Epoch 5 loss 0.37215637943427105 valid acc 15/16
Epoch 5 loss 1.3014367560188245 valid acc 16/16
Epoch 5 loss 1.2385163481563257 valid acc 16/16
Epoch 5 loss 1.4177338870762892 valid acc 14/16
Epoch 5 loss 1.1060807470443776 valid acc 15/16
Epoch 5 loss 0.6847725839791202 valid acc 16/16
Epoch 5 loss 1.2859637604831038 valid acc 14/16
Epoch 5 loss 0.7685204641190525 valid acc 15/16
Epoch 5 loss 0.45956303241749696 valid acc 14/16
Epoch 5 loss 1.389271482008462 valid acc 14/16
Epoch 5 loss 1.456795279164861 valid acc 14/16
Epoch 5 loss 0.8356907786921945 valid acc 15/16
Epoch 5 loss 0.2152679805360023 valid acc 14/16
Epoch 5 loss 0.4473308197445519 valid acc 14/16
Epoch 5 loss 1.123008593702301 valid acc 14/16
Epoch 5 loss 1.329186649369448 valid acc 16/16
Epoch 5 loss 0.46804758293258625 valid acc 16/16
Epoch 5 loss 0.5083375613155485 valid acc 15/16
Epoch 5 loss 0.7213687960773152 valid acc 14/16
Epoch 5 loss 0.44560213021132766 valid acc 16/16
Epoch 5 loss 0.6541591806756154 valid acc 15/16
Epoch 5 loss 0.5982562641809526 valid acc 14/16
Epoch 5 loss 0.1918303768286269 valid acc 14/16
Epoch 5 loss 0.9323507795721049 valid acc 16/16
Epoch 5 loss 1.3843829335313589 valid acc 16/16
Epoch 5 loss 0.8935946104408103 valid acc 14/16
Epoch 5 loss 0.7011827192996256 valid acc 14/16
Epoch 5 loss 0.667805830224343 valid acc 14/16
Epoch 5 loss 0.43538483948609275 valid acc 15/16
Epoch 5 loss 0.4778995034864593 valid acc 14/16
Epoch 5 loss 0.6093177367817213 valid acc 14/16
Epoch 5 loss 0.6228762116395146 valid acc 15/16
Epoch 5 loss 0.7555127258286112 valid acc 15/16
Epoch 5 loss 0.585654296221584 valid acc 16/16
Epoch 5 loss 1.0114318611716242 valid acc 16/16
Epoch 5 loss 0.3451127319601117 valid acc 16/16
Epoch 5 loss 0.5033858759929914 valid acc 16/16
Epoch 5 loss 1.3047275774621823 valid acc 15/16
Epoch 5 loss 0.4656137296310422 valid acc 15/16
Epoch 5 loss 0.3283223377658031 valid acc 15/16
Epoch 5 loss 0.21779351110285033 valid acc 16/16
Epoch 5 loss 0.5929444020279371 valid acc 15/16
Epoch 5 loss 0.9218769572858998 valid acc 16/16
Epoch 5 loss 0.3459063888911914 valid acc 16/16
Epoch 5 loss 0.7809780380456284 valid acc 13/16
Epoch 5 loss 0.5867428482425537 valid acc 15/16
Epoch 5 loss 0.8939373611842292 valid acc 16/16
Epoch 5 loss 0.6965759722238825 valid acc 16/16
Epoch 5 loss 0.7367310089177073 valid acc 16/16
Epoch 5 loss 0.7100737379826039 valid acc 16/16
Epoch 5 loss 0.8197827316577209 valid acc 16/16
Epoch 5 loss 0.4198549395160407 valid acc 16/16
Epoch 5 loss 0.4263904756706114 valid acc 15/16
Epoch 6 loss 0.007267793362282671 valid acc 16/16
Epoch 6 loss 0.5541983116405939 valid acc 16/16
Epoch 6 loss 0.8844311605071769 valid acc 16/16
Epoch 6 loss 0.9058534726741896 valid acc 16/16
Epoch 6 loss 0.5518447079761784 valid acc 16/16
Epoch 6 loss 0.9156752015850498 valid acc 16/16
Epoch 6 loss 0.7366943062646791 valid acc 16/16
Epoch 6 loss 1.0481523034227505 valid acc 16/16
Epoch 6 loss 0.3556347762686084 valid acc 16/16
Epoch 6 loss 0.27335412551412586 valid acc 16/16
Epoch 6 loss 0.5468303706459309 valid acc 16/16
Epoch 6 loss 1.1324649562080233 valid acc 16/16
Epoch 6 loss 1.090553048284898 valid acc 16/16
Epoch 6 loss 1.5043972710751636 valid acc 15/16
Epoch 6 loss 0.9030343069189555 valid acc 15/16
Epoch 6 loss 0.5810988206795726 valid acc 15/16
Epoch 6 loss 1.2462172632503292 valid acc 15/16
Epoch 6 loss 0.30497232638742283 valid acc 16/16
Epoch 6 loss 0.8706933898871593 valid acc 15/16
Epoch 6 loss 0.34126230596761037 valid acc 15/16
Epoch 6 loss 1.1513649694458112 valid acc 15/16
Epoch 6 loss 0.5368761604823449 valid acc 15/16
Epoch 6 loss 0.1304239040120902 valid acc 15/16
Epoch 6 loss 0.41908776868104625 valid acc 15/16
Epoch 6 loss 0.41908871165350065 valid acc 14/16
Epoch 6 loss 0.45304370709736236 valid acc 15/16
Epoch 6 loss 0.17674799385372664 valid acc 16/16
Epoch 6 loss 0.516064225595007 valid acc 15/16
Epoch 6 loss 0.09915818620480926 valid acc 15/16
Epoch 6 loss 0.11592334752091776 valid acc 16/16
Epoch 6 loss 0.17781299653152272 valid acc 16/16
Epoch 6 loss 0.9192270279751096 valid acc 16/16
Epoch 6 loss 0.12363918522726991 valid acc 16/16
Epoch 6 loss 1.1677319329777225 valid acc 16/16
Epoch 6 loss 1.3664808438049634 valid acc 16/16
Epoch 6 loss 0.47048855527831546 valid acc 16/16
Epoch 6 loss 0.4987322668000849 valid acc 15/16
Epoch 6 loss 0.3304677042475865 valid acc 16/16
Epoch 6 loss 0.6648909805932448 valid acc 15/16
Epoch 6 loss 0.9219079517361511 valid acc 15/16
Epoch 6 loss 0.460540199949756 valid acc 16/16
Epoch 6 loss 0.8964735530294611 valid acc 15/16
Epoch 6 loss 0.2828573604437648 valid acc 16/16
Epoch 6 loss 0.11108911126267451 valid acc 16/16
Epoch 6 loss 1.5263480214744511 valid acc 16/16
Epoch 6 loss 0.1591960919345503 valid acc 16/16
Epoch 6 loss 0.7049277164416572 valid acc 16/16
Epoch 6 loss 0.7261860992682695 valid acc 16/16
Epoch 6 loss 0.48245839167732885 valid acc 15/16
Epoch 6 loss 0.38670240427093105 valid acc 15/16
Epoch 6 loss 0.6021529613401385 valid acc 14/16
Epoch 6 loss 0.45312640948502114 valid acc 16/16
Epoch 6 loss 0.2660124058385379 valid acc 15/16
Epoch 6 loss 0.3538678125122561 valid acc 15/16
Epoch 6 loss 0.43134844007645345 valid acc 15/16
Epoch 6 loss 0.16626831020053534 valid acc 15/16
Epoch 6 loss 0.3812636235873719 valid acc 15/16
Epoch 6 loss 0.3785562765395134 valid acc 16/16
Epoch 6 loss 1.0710140620652582 valid acc 15/16
Epoch 6 loss 0.816765882981306 valid acc 15/16
Epoch 6 loss 1.0285764413343164 valid acc 16/16
Epoch 6 loss 0.5636412073203267 valid acc 16/16
Epoch 6 loss 0.6291337059818578 valid acc 15/16
Epoch 7 loss 0.010377027376322734 valid acc 16/16
Epoch 7 loss 0.7655122660786082 valid acc 15/16
Epoch 7 loss 0.6912604140871184 valid acc 16/16
Epoch 7 loss 0.3816340303124929 valid acc 16/16
Epoch 7 loss 0.18798563852444256 valid acc 16/16
Epoch 7 loss 0.27688671337247245 valid acc 16/16
Epoch 7 loss 0.6772563580010684 valid acc 16/16
Epoch 7 loss 0.6578327105779029 valid acc 16/16
Epoch 7 loss 0.35012588938294403 valid acc 16/16
Epoch 7 loss 0.3853658301369076 valid acc 15/16
Epoch 7 loss 0.30650667971803564 valid acc 15/16
Epoch 7 loss 1.097350028988882 valid acc 14/16
Epoch 7 loss 0.9520553805380668 valid acc 16/16
Epoch 7 loss 0.6159455667370803 valid acc 15/16
Epoch 7 loss 0.31886948761529826 valid acc 15/16
Epoch 7 loss 0.2884813639226994 valid acc 14/16
Epoch 7 loss 0.8963807798952258 valid acc 15/16
Epoch 7 loss 0.5741241770019458 valid acc 15/16
Epoch 7 loss 0.7338979586522968 valid acc 16/16
Epoch 7 loss 0.40330906444885445 valid acc 15/16
Epoch 7 loss 0.6862803299080179 valid acc 13/16
Epoch 7 loss 0.750233276974331 valid acc 15/16
Epoch 7 loss 0.5877710208849283 valid acc 16/16
Epoch 7 loss 0.4746465227767212 valid acc 15/16
Epoch 7 loss 0.7171868563209953 valid acc 15/16
Epoch 7 loss 0.4078382056896273 valid acc 15/16
Epoch 7 loss 0.14235341418309727 valid acc 15/16
Epoch 7 loss 0.3904932871553874 valid acc 15/16
Epoch 7 loss 0.3482085554603891 valid acc 15/16
Epoch 7 loss 0.14926003337639843 valid acc 15/16
Epoch 7 loss 0.3945420173529105 valid acc 15/16
Epoch 7 loss 0.5111430638904799 valid acc 14/16
Epoch 7 loss 0.22122089247378568 valid acc 13/16
Epoch 7 loss 0.6232043953894614 valid acc 16/16
Epoch 7 loss 0.7094266089052922 valid acc 16/16
Epoch 7 loss 1.2148723215939845 valid acc 15/16
Epoch 7 loss 0.4034894083063343 valid acc 15/16
Epoch 7 loss 0.2919618356563285 valid acc 15/16
Epoch 7 loss 0.2053178271155341 valid acc 15/16
Epoch 7 loss 0.4103606480869031 valid acc 16/16
Epoch 7 loss 0.23053843901159382 valid acc 15/16
Epoch 7 loss 0.26649807229195865 valid acc 16/16
Epoch 7 loss 0.37791177124517794 valid acc 16/16
Epoch 7 loss 0.19767047366071258 valid acc 16/16
Epoch 7 loss 1.0245336480403913 valid acc 16/16
Epoch 7 loss 0.2361317611051374 valid acc 15/16
Epoch 7 loss 1.0810152913396545 valid acc 16/16
Epoch 7 loss 0.7669524824950823 valid acc 15/16
Epoch 7 loss 0.20352853999638781 valid acc 15/16
Epoch 7 loss 0.0940540002424643 valid acc 16/16
Epoch 7 loss 0.42401955777339057 valid acc 16/16
Epoch 7 loss 0.6157997198096179 valid acc 15/16
Epoch 7 loss 0.9614530073955713 valid acc 16/16
Epoch 7 loss 0.15752541831474387 valid acc 16/16
Epoch 7 loss 0.7912366074464539 valid acc 16/16
Epoch 7 loss 0.225465211925782 valid acc 16/16
Epoch 7 loss 0.26876525522893124 valid acc 16/16
Epoch 7 loss 0.39335431255255177 valid acc 15/16
Epoch 7 loss 0.6171092943182567 valid acc 15/16
Epoch 7 loss 0.8515657413185964 valid acc 16/16
Epoch 7 loss 0.787453615783871 valid acc 14/16
Epoch 7 loss 0.6079198856203933 valid acc 16/16
Epoch 7 loss 1.0760280240957867 valid acc 15/16
Epoch 8 loss 0.007571565366244037 valid acc 15/16
Epoch 8 loss 0.5277543637472908 valid acc 15/16
Epoch 8 loss 0.5545305500609355 valid acc 15/16
Epoch 8 loss 1.3486674984422327 valid acc 15/16
Epoch 8 loss 0.26054559280690903 valid acc 13/16
Epoch 8 loss 0.49711125608477846 valid acc 15/16
Epoch 8 loss 0.6502615521714767 valid acc 15/16
Epoch 8 loss 0.4129096940929413 valid acc 15/16
Epoch 8 loss 1.1127816217134465 valid acc 16/16
Epoch 8 loss 0.11583134728457903 valid acc 15/16
Epoch 8 loss 0.34013504165551717 valid acc 16/16
Epoch 8 loss 1.122648267793225 valid acc 15/16
Epoch 8 loss 1.4872219194392489 valid acc 16/16
Epoch 8 loss 1.1422310130568287 valid acc 16/16
Epoch 8 loss 0.40095759762925276 valid acc 16/16
Epoch 8 loss 0.5886597851603645 valid acc 16/16
Epoch 8 loss 0.4282121300935162 valid acc 16/16
Epoch 8 loss 0.306447567320694 valid acc 16/16
Epoch 8 loss 0.13573074564281168 valid acc 16/16
Epoch 8 loss 0.24113982365130257 valid acc 15/16
Epoch 8 loss 0.5479831713801404 valid acc 16/16
Epoch 8 loss 0.311619179939283 valid acc 15/16
Epoch 8 loss 0.06287357781684516 valid acc 15/16
Epoch 8 loss 0.11475424999588879 valid acc 15/16
Epoch 8 loss 0.5594906607636387 valid acc 15/16
Epoch 8 loss 0.517442184845308 valid acc 15/16
Epoch 8 loss 0.8521724119966988 valid acc 15/16
Epoch 8 loss 0.20505842408996133 valid acc 15/16
Epoch 8 loss 0.5583679261491459 valid acc 15/16
Epoch 8 loss 0.3178814201606868 valid acc 15/16
Epoch 8 loss 0.2604843321609392 valid acc 16/16
Epoch 8 loss 0.9891355349747524 valid acc 15/16
Epoch 8 loss 0.2155957306174364 valid acc 15/16
Epoch 8 loss 0.9763029479324095 valid acc 15/16
Epoch 8 loss 0.8321090765744654 valid acc 16/16
Epoch 8 loss 0.6199183453616097 valid acc 16/16
Epoch 8 loss 0.23725982711205026 valid acc 15/16
Epoch 8 loss 0.6297108875114483 valid acc 16/16
Epoch 8 loss 0.9187749333529326 valid acc 15/16
Epoch 8 loss 0.4500645044869588 valid acc 16/16
Epoch 8 loss 0.5119003150972539 valid acc 14/16
Epoch 8 loss 0.3377631052832999 valid acc 15/16
Epoch 8 loss 0.33816406904830554 valid acc 15/16
Epoch 8 loss 0.20748440719067346 valid acc 16/16
Epoch 8 loss 0.7925801519059155 valid acc 16/16
Epoch 8 loss 0.17959095199539624 valid acc 16/16
Epoch 8 loss 0.2282427135633196 valid acc 16/16
Epoch 8 loss 0.6780856994977604 valid acc 16/16
Epoch 8 loss 0.21303646377818136 valid acc 16/16
Epoch 8 loss 0.05633011593433512 valid acc 16/16
Epoch 8 loss 0.10603799673970787 valid acc 16/16
Epoch 8 loss 0.4801973283055783 valid acc 16/16
Epoch 8 loss 0.24277378324600052 valid acc 16/16
Epoch 8 loss 0.6817608304288332 valid acc 16/16
Epoch 8 loss 0.34510343989024245 valid acc 16/16
Epoch 8 loss 0.09847541876553895 valid acc 16/16
Epoch 8 loss 0.28521254567671533 valid acc 15/16
Epoch 8 loss 0.21452078795317708 valid acc 16/16
Epoch 8 loss 0.5835481976840077 valid acc 16/16
Epoch 8 loss 0.3233146920534255 valid acc 16/16
Epoch 8 loss 0.32914488815423043 valid acc 16/16
Epoch 8 loss 0.2136265348419311 valid acc 16/16
Epoch 8 loss 0.767653551300525 valid acc 15/16
Epoch 9 loss 0.014372565173369045 valid acc 14/16
Epoch 9 loss 0.45299039569909244 valid acc 14/16
Epoch 9 loss 0.4751448807079169 valid acc 14/16
Epoch 9 loss 0.2684612978995484 valid acc 14/16
Epoch 9 loss 0.04080437625997768 valid acc 14/16
Epoch 9 loss 0.16393704674080936 valid acc 15/16
Epoch 9 loss 0.46069186614161906 valid acc 15/16
Epoch 9 loss 0.3743993869223752 valid acc 16/16
Epoch 9 loss 0.38648453682709144 valid acc 15/16
Epoch 9 loss 0.11568667063222277 valid acc 14/16
Epoch 9 loss 0.2565179963184108 valid acc 16/16
Epoch 9 loss 0.6682316502910256 valid acc 15/16
Epoch 9 loss 0.2478144001714344 valid acc 14/16
Epoch 9 loss 1.353943733971383 valid acc 14/16
Epoch 9 loss 0.5054876050636621 valid acc 14/16
Epoch 9 loss 0.11557588326899401 valid acc 15/16
Epoch 9 loss 0.8482103469641079 valid acc 15/16
Epoch 9 loss 0.44170147461171005 valid acc 15/16
Epoch 9 loss 0.7076727822164715 valid acc 15/16
Epoch 9 loss 0.2737326793294053 valid acc 15/16
Epoch 9 loss 0.30785121644157076 valid acc 14/16
Epoch 9 loss 0.1974201529811883 valid acc 15/16
Epoch 9 loss 0.1510722994769731 valid acc 15/16
Epoch 9 loss 0.23074340533575355 valid acc 15/16
Epoch 9 loss 0.30899095143555017 valid acc 15/16
Epoch 9 loss 0.5136132327401668 valid acc 15/16
Epoch 9 loss 0.2735611470180027 valid acc 16/16
Epoch 9 loss 0.21832922514770375 valid acc 16/16
Epoch 9 loss 0.41449078688459423 valid acc 15/16
Epoch 9 loss 0.7485241540486973 valid acc 15/16
Epoch 9 loss 0.26465691440884115 valid acc 16/16
Epoch 9 loss 0.6272516858353061 valid acc 16/16
Epoch 9 loss 0.16243290089521756 valid acc 16/16
Epoch 9 loss 0.1260958888227558 valid acc 16/16
Epoch 9 loss 0.7238305645590533 valid acc 15/16
Epoch 9 loss 0.6893636634909251 valid acc 15/16
Epoch 9 loss 0.15542796887460186 valid acc 15/16
Epoch 9 loss 0.2733560031585028 valid acc 15/16
Epoch 9 loss 0.052618111361990016 valid acc 15/16
Epoch 9 loss 0.2648647152807537 valid acc 16/16
Epoch 9 loss 0.18657205002346072 valid acc 16/16
Epoch 9 loss 0.27431266873452786 valid acc 16/16
Epoch 9 loss 0.3497165898049271 valid acc 16/16
Epoch 9 loss 0.5857787499910708 valid acc 15/16
Epoch 9 loss 0.7182810657164589 valid acc 15/16
Epoch 9 loss 0.25701844467921464 valid acc 16/16
Epoch 9 loss 0.12282880816920011 valid acc 16/16
Epoch 9 loss 0.4325939944121243 valid acc 16/16
Epoch 9 loss 0.13660806530714978 valid acc 15/16
Epoch 9 loss 0.0630397407980966 valid acc 15/16
Epoch 9 loss 0.04465355766760071 valid acc 16/16
Epoch 9 loss 0.18889087862714915 valid acc 15/16
Epoch 9 loss 0.48981587650104313 valid acc 15/16
Epoch 9 loss 0.023968983987410097 valid acc 15/16
Epoch 9 loss 0.31612777578953305 valid acc 16/16
Epoch 9 loss 0.19130665333379387 valid acc 16/16
Epoch 9 loss 1.0964991689987529 valid acc 14/16
Epoch 9 loss 0.17281538950377853 valid acc 15/16
Epoch 9 loss 0.38228997410564053 valid acc 15/16
Epoch 9 loss 0.5503920418850371 valid acc 16/16
Epoch 9 loss 0.8756368735727602 valid acc 16/16
Epoch 9 loss 0.9414803370493929 valid acc 16/16
Epoch 9 loss 0.4240042319046302 valid acc 16/16
Epoch 10 loss 0.029146908900414492 valid acc 16/16
Epoch 10 loss 0.3494014824889744 valid acc 16/16
Epoch 10 loss 0.2395293287748633 valid acc 16/16
Epoch 10 loss 0.672291309648047 valid acc 14/16
Epoch 10 loss 0.4274554350461992 valid acc 15/16
Epoch 10 loss 0.5197555247772644 valid acc 16/16
Epoch 10 loss 0.3230832528338909 valid acc 16/16
Epoch 10 loss 0.8050876902098795 valid acc 16/16
Epoch 10 loss 0.5522717451804542 valid acc 16/16
Epoch 10 loss 0.16200488469068092 valid acc 16/16
Epoch 10 loss 0.5356512181070916 valid acc 16/16
Epoch 10 loss 0.3318821488973586 valid acc 16/16
Epoch 10 loss 0.230922489647788 valid acc 16/16
Epoch 10 loss 0.36888688643861833 valid acc 16/16
Epoch 10 loss 0.3358284261387274 valid acc 16/16
Epoch 10 loss 0.36938099947048747 valid acc 16/16
Epoch 10 loss 0.7646912266741261 valid acc 16/16
Epoch 10 loss 0.14195951842103716 valid acc 16/16
Epoch 10 loss 0.7813516922538433 valid acc 16/16
Epoch 10 loss 0.2404245503210585 valid acc 16/16
Epoch 10 loss 0.6033235174715825 valid acc 16/16
Epoch 10 loss 0.3069134417881849 valid acc 16/16
Epoch 10 loss 0.02368422298027162 valid acc 16/16
Epoch 10 loss 0.23397545164032424 valid acc 16/16
Epoch 10 loss 0.6784059756516618 valid acc 16/16
Epoch 10 loss 0.6173074409235862 valid acc 16/16
Epoch 10 loss 0.2283018381574689 valid acc 16/16
Epoch 10 loss 0.12785794325364352 valid acc 16/16
Epoch 10 loss 0.04987586825489598 valid acc 15/16
Epoch 10 loss 0.2746788825755526 valid acc 16/16
Epoch 10 loss 0.10213075849077335 valid acc 16/16
Epoch 10 loss 0.7054462127520695 valid acc 16/16
Epoch 10 loss 0.20012875490771248 valid acc 15/16
Epoch 10 loss 0.20131463687398599 valid acc 16/16
Epoch 10 loss 1.0558480258127751 valid acc 16/16
Epoch 10 loss 0.675273642182891 valid acc 16/16
Epoch 10 loss 0.35694393283034337 valid acc 16/16
Epoch 10 loss 0.27000006092589046 valid acc 16/16
Epoch 10 loss 0.12920397240957793 valid acc 16/16
Epoch 10 loss 0.15255606209048242 valid acc 16/16
Epoch 10 loss 0.3106389026429054 valid acc 16/16
Epoch 10 loss 0.1276972539238131 valid acc 16/16
Epoch 10 loss 1.037289112136526 valid acc 16/16
Epoch 10 loss 0.2825195243893899 valid acc 16/16
Epoch 10 loss 0.8319690709305989 valid acc 16/16
Epoch 10 loss 0.2906501196869725 valid acc 16/16
Epoch 10 loss 0.19891985214972713 valid acc 16/16
Epoch 10 loss 0.24244488845065926 valid acc 16/16
Epoch 10 loss 0.4497495989269825 valid acc 16/16
Epoch 10 loss 0.11451237059726727 valid acc 16/16
Epoch 10 loss 0.3725072852345989 valid acc 16/16
Epoch 10 loss 0.10366364749098866 valid acc 16/16
Epoch 10 loss 0.2569089735065428 valid acc 16/16
Epoch 10 loss 0.2765898104670043 valid acc 16/16
Epoch 10 loss 0.4088716317214278 valid acc 16/16
Epoch 10 loss 0.11696215695606593 valid acc 16/16
Epoch 10 loss 0.2973762996821859 valid acc 16/16
Epoch 10 loss 0.16764562811856967 valid acc 16/16
Epoch 10 loss 0.4349638430832982 valid acc 16/16
Epoch 10 loss 0.17876982722947088 valid acc 16/16
Epoch 10 loss 0.4261185539458116 valid acc 16/16
Epoch 10 loss 0.8507007654340754 valid acc 16/16
Epoch 10 loss 0.24177419391258576 valid acc 16/16
Epoch 11 loss 0.0418350489216453 valid acc 16/16
Epoch 11 loss 0.4322526054103619 valid acc 16/16
Epoch 11 loss 0.5606841465492061 valid acc 16/16
Epoch 11 loss 0.2242084933888751 valid acc 16/16
Epoch 11 loss 0.141422714099872 valid acc 16/16
Epoch 11 loss 0.28197073507846343 valid acc 15/16
Epoch 11 loss 0.0931528081713684 valid acc 15/16
Epoch 11 loss 0.5587717792409979 valid acc 16/16
Epoch 11 loss 0.20435871518312393 valid acc 16/16
Epoch 11 loss 0.10534043363152357 valid acc 16/16
Epoch 11 loss 0.30527054575746426 valid acc 16/16
Epoch 11 loss 0.2108700098618119 valid acc 16/16
Epoch 11 loss 0.520692717890079 valid acc 16/16
Epoch 11 loss 0.8660933048591901 valid acc 16/16
Epoch 11 loss 0.15297417317927592 valid acc 16/16
Epoch 11 loss 0.3276333191980532 valid acc 16/16
Epoch 11 loss 0.3929839529609428 valid acc 15/16
Epoch 11 loss 0.3892887698897971 valid acc 16/16
Epoch 11 loss 0.532169146966142 valid acc 15/16
Epoch 11 loss 0.29012205124526946 valid acc 16/16
Epoch 11 loss 0.2541889325451525 valid acc 15/16
Epoch 11 loss 0.4427560537150915 valid acc 14/16
Epoch 11 loss 0.16009995876851146 valid acc 16/16
Epoch 11 loss 0.2584772160060306 valid acc 14/16
Epoch 11 loss 0.311673002519724 valid acc 15/16
Epoch 11 loss 0.22100148194113112 valid acc 15/16
Epoch 11 loss 0.33331338543076555 valid acc 15/16
Epoch 11 loss 0.28181038184336243 valid acc 14/16
Epoch 11 loss 0.16018063514494002 valid acc 15/16
Epoch 11 loss 0.24962584575338964 valid acc 15/16
Epoch 11 loss 0.5972755503527307 valid acc 15/16
Epoch 11 loss 1.0040349788588478 valid acc 15/16
Epoch 11 loss 0.20427227104973755 valid acc 15/16
Epoch 11 loss 0.07017490408182792 valid acc 15/16
Epoch 11 loss 0.6414429249477006 valid acc 16/16
Epoch 11 loss 0.47003111680888665 valid acc 16/16
Epoch 11 loss 0.43354557862000076 valid acc 16/16
Epoch 11 loss 0.5367790003475452 valid acc 16/16
Epoch 11 loss 0.12913651172756302 valid acc 16/16
Epoch 11 loss 0.1916555617543057 valid acc 16/16
Epoch 11 loss 0.15815250425299637 valid acc 15/16
Epoch 11 loss 0.2422184778634428 valid acc 16/16
Epoch 11 loss 0.33615781254308746 valid acc 15/16
Epoch 11 loss 0.04809838315061887 valid acc 15/16
Epoch 11 loss 0.47192109363022827 valid acc 16/16
Epoch 11 loss 0.8258690136994274 valid acc 16/16
Epoch 11 loss 0.17511217413510943 valid acc 16/16
Epoch 11 loss 0.28432120728467813 valid acc 16/16
Epoch 11 loss 0.7445679207062216 valid acc 15/16
Epoch 11 loss 1.014462672821009 valid acc 16/16
Epoch 11 loss 0.2040817402388574 valid acc 16/16
Epoch 11 loss 1.574025090913518 valid acc 16/16
Epoch 11 loss 0.32081972684289267 valid acc 16/16
Epoch 11 loss 0.08074556594014082 valid acc 16/16
Epoch 11 loss 0.5220100786297988 valid acc 14/16
Epoch 11 loss 0.25371453825556567 valid acc 15/16
Epoch 11 loss 0.8179038183791628 valid acc 16/16
Epoch 11 loss 0.18857109761073282 valid acc 16/16
Epoch 11 loss 1.3524548824658715 valid acc 16/16
Epoch 11 loss 0.799778718274411 valid acc 16/16
Epoch 11 loss 0.3026523306882132 valid acc 15/16
Epoch 11 loss 0.2392012269689286 valid acc 15/16
Epoch 11 loss 0.23598223203369406 valid acc 15/16
Epoch 12 loss 0.0019502783752319286 valid acc 15/16
Epoch 12 loss 0.5181968141118153 valid acc 15/16
Epoch 12 loss 0.41696363515542545 valid acc 16/16
Epoch 12 loss 0.46810368614803693 valid acc 16/16
Epoch 12 loss 0.26584257103656006 valid acc 15/16
Epoch 12 loss 0.1569382144922581 valid acc 16/16
Epoch 12 loss 0.44574326385908913 valid acc 15/16
Epoch 12 loss 0.7601584308729481 valid acc 16/16
Epoch 12 loss 0.33676671981772166 valid acc 16/16
Epoch 12 loss 0.09817133174827097 valid acc 16/16
Epoch 12 loss 0.43465967274197126 valid acc 15/16
Epoch 12 loss 0.37668108669693134 valid acc 16/16
Epoch 12 loss 0.1380417473106078 valid acc 16/16
Epoch 12 loss 0.8371736522396349 valid acc 15/16
Epoch 12 loss 0.3689094747382793 valid acc 16/16
Epoch 12 loss 0.2362677574538948 valid acc 16/16
Epoch 12 loss 0.5864966889818805 valid acc 16/16
Epoch 12 loss 0.5222208485155823 valid acc 16/16
Epoch 12 loss 0.26813299639818894 valid acc 16/16
Epoch 12 loss 0.4950284009057711 valid acc 16/16
Epoch 12 loss 0.4613194921985592 valid acc 16/16
Epoch 12 loss 0.2922673287123694 valid acc 16/16
Epoch 12 loss 0.022853157313732386 valid acc 16/16
Epoch 12 loss 1.408097303338745 valid acc 14/16
Epoch 12 loss 0.32235401916726975 valid acc 15/16
Epoch 12 loss 0.5227292653518067 valid acc 15/16
Epoch 12 loss 0.19777050182507763 valid acc 16/16
Epoch 12 loss 0.2268882397762983 valid acc 15/16
Epoch 12 loss 0.11300540689514132 valid acc 15/16
Epoch 12 loss 0.059479852699509894 valid acc 15/16
Epoch 12 loss 0.301370177201651 valid acc 15/16
Epoch 12 loss 0.22768163633123184 valid acc 16/16
Epoch 12 loss 0.08973229440169977 valid acc 15/16
Epoch 12 loss 0.11889812327480309 valid acc 16/16
Epoch 12 loss 0.6723332779217119 valid acc 16/16
Epoch 12 loss 0.5222256386808861 valid acc 15/16
Epoch 12 loss 0.546201890033213 valid acc 15/16
Epoch 12 loss 0.025081701815726776 valid acc 15/16
Epoch 12 loss 0.14249748990436711 valid acc 16/16
Epoch 12 loss 0.44079826024727137 valid acc 16/16
Epoch 12 loss 0.15345847873491306 valid acc 16/16
Epoch 12 loss 0.810962367948021 valid acc 16/16
Epoch 12 loss 0.12702821068485515 valid acc 16/16
Epoch 12 loss 0.05321880439118293 valid acc 16/16
Epoch 12 loss 0.5954938202798391 valid acc 16/16
Epoch 12 loss 0.02454816498861645 valid acc 16/16
Epoch 12 loss 0.26259559104557134 valid acc 16/16
Epoch 12 loss 0.5885832133815716 valid acc 16/16
Epoch 12 loss 0.15364689774366191 valid acc 16/16
Epoch 12 loss 0.0339869301064461 valid acc 16/16
Epoch 12 loss 0.22145525516083328 valid acc 16/16
Epoch 12 loss 0.029356327351033384 valid acc 16/16
Epoch 12 loss 0.18357314425249555 valid acc 16/16
Epoch 12 loss 0.1537915172760173 valid acc 16/16
Epoch 12 loss 0.206680816402933 valid acc 16/16
Epoch 12 loss 0.02320117193785798 valid acc 16/16
Epoch 12 loss 0.39241757171143204 valid acc 15/16
Epoch 12 loss 0.20840981325796532 valid acc 16/16
Epoch 12 loss 0.06481380424733624 valid acc 16/16
Epoch 12 loss 0.2855268284765277 valid acc 16/16
Epoch 12 loss 0.18725098469815638 valid acc 16/16
Epoch 12 loss 0.026961317161220466 valid acc 16/16
Epoch 12 loss 0.14484620852886662 valid acc 16/16
Epoch 13 loss 0.00045946839928645566 valid acc 16/16
Epoch 13 loss 0.6131112328023531 valid acc 16/16
Epoch 13 loss 0.7551393949272167 valid acc 15/16
Epoch 13 loss 0.42239799876476725 valid acc 15/16
Epoch 13 loss 0.25093093442414677 valid acc 15/16
Epoch 13 loss 0.21972346516202604 valid acc 15/16
Epoch 13 loss 1.035258898447338 valid acc 15/16
Epoch 13 loss 0.568231311371699 valid acc 15/16
Epoch 13 loss 0.12228854822524704 valid acc 15/16
Epoch 13 loss 0.2334642300941999 valid acc 15/16
Epoch 13 loss 0.3096312494441833 valid acc 15/16
Epoch 13 loss 0.17798481302127894 valid acc 15/16
Epoch 13 loss 0.09644649959071683 valid acc 15/16
Epoch 13 loss 0.43308698331519363 valid acc 15/16
Epoch 13 loss 0.5058453125889788 valid acc 16/16
Epoch 13 loss 0.28004789070794006 valid acc 16/16
Epoch 13 loss 0.8027651117248377 valid acc 16/16
Epoch 13 loss 0.21910237954908118 valid acc 16/16
Epoch 13 loss 0.47828635202684194 valid acc 16/16
Epoch 13 loss 0.5456637338514375 valid acc 16/16
Epoch 13 loss 0.15208769601548855 valid acc 16/16
Epoch 13 loss 0.44018014338091277 valid acc 16/16
Epoch 13 loss 0.030155899328395297 valid acc 16/16
Epoch 13 loss 0.06891786245435333 valid acc 16/16
Epoch 13 loss 0.32870174158249815 valid acc 15/16
Epoch 13 loss 0.4057883450728188 valid acc 15/16
Epoch 13 loss 0.4780625642164394 valid acc 15/16
Epoch 13 loss 0.05733931598569236 valid acc 15/16
Epoch 13 loss 0.2688294425678639 valid acc 15/16
Epoch 13 loss 0.10199185956131962 valid acc 15/16
Epoch 13 loss 0.2717836999010643 valid acc 15/16
Epoch 13 loss 0.6328875138277181 valid acc 15/16
Epoch 13 loss 0.19650368445209015 valid acc 15/16
Epoch 13 loss 0.38665932068199166 valid acc 16/16
Epoch 13 loss 0.35100503408269884 valid acc 16/16
Epoch 13 loss 1.095924193688184 valid acc 16/16
Epoch 13 loss 0.3236711753635518 valid acc 15/16
Epoch 13 loss 0.9550942608275357 valid acc 16/16
Epoch 13 loss 0.2318078454646436 valid acc 16/16
Epoch 13 loss 0.1167362863982076 valid acc 15/16
Epoch 13 loss 0.24188749589859584 valid acc 15/16
Epoch 13 loss 0.11070970518496276 valid acc 16/16
Epoch 13 loss 0.23386543983046199 valid acc 15/16
Epoch 13 loss 0.6340230993475837 valid acc 16/16
Epoch 13 loss 0.7406404180131709 valid acc 16/16
Epoch 13 loss 0.1524829747262454 valid acc 16/16
Epoch 13 loss 0.08618515020270928 valid acc 16/16
Epoch 13 loss 0.5779370132450558 valid acc 16/16
Epoch 13 loss 0.17549301699406156 valid acc 16/16
Epoch 13 loss 0.1589523412272117 valid acc 16/16
Epoch 13 loss 0.11995263443765714 valid acc 16/16
Epoch 13 loss 0.1502779250477918 valid acc 16/16
Epoch 13 loss 0.10345808163763626 valid acc 16/16
Epoch 13 loss 0.1523629900239087 valid acc 16/16
Epoch 13 loss 0.3577908250470845 valid acc 16/16
Epoch 13 loss 0.1266244454594173 valid acc 16/16
Epoch 13 loss 0.7853391121501706 valid acc 16/16
Epoch 13 loss 0.5867777004257803 valid acc 16/16
Epoch 13 loss 0.23278284712998826 valid acc 16/16
Epoch 13 loss 0.0740613270144887 valid acc 16/16
Epoch 13 loss 0.07229075879605606 valid acc 16/16
Epoch 13 loss 0.1692270077089512 valid acc 16/16
Epoch 13 loss 0.9926876811994005 valid acc 16/16
Epoch 14 loss 0.036517287338408755 valid acc 16/16
Epoch 14 loss 0.3411417462416245 valid acc 16/16
Epoch 14 loss 0.4686257474095183 valid acc 16/16
Epoch 14 loss 0.14439855675132268 valid acc 16/16
Epoch 14 loss 0.05337655035808064 valid acc 16/16
Epoch 14 loss 0.2490014309112099 valid acc 16/16
Epoch 14 loss 0.2498318333508046 valid acc 16/16
Epoch 14 loss 0.36860548123687137 valid acc 16/16
Epoch 14 loss 0.3274983562322116 valid acc 16/16
Epoch 14 loss 0.3101811267228436 valid acc 16/16
Epoch 14 loss 0.19464982010085394 valid acc 16/16
Epoch 14 loss 0.3355236426352999 valid acc 16/16
Epoch 14 loss 0.27080477254181723 valid acc 16/16
Epoch 14 loss 0.6062461347305654 valid acc 16/16
Epoch 14 loss 0.16507089864601746 valid acc 16/16
Epoch 14 loss 0.05102269761239059 valid acc 16/16
Epoch 14 loss 0.21489547474323292 valid acc 16/16
Epoch 14 loss 0.5277474272783194 valid acc 16/16
Epoch 14 loss 0.2212604423520116 valid acc 16/16
Epoch 14 loss 0.10381300854079673 valid acc 16/16
Epoch 14 loss 0.4828651478997236 valid acc 16/16
Epoch 14 loss 0.1563983819851572 valid acc 16/16
Epoch 14 loss 0.008596207666445643 valid acc 16/16
Epoch 14 loss 0.5789158098174068 valid acc 16/16
Epoch 14 loss 0.13811386057466013 valid acc 16/16
Epoch 14 loss 0.49848963877222374 valid acc 16/16
Epoch 14 loss 0.43593198924711046 valid acc 16/16
Epoch 14 loss 0.16604990630566419 valid acc 16/16
Epoch 14 loss 0.0595321182976003 valid acc 16/16
Epoch 14 loss 0.4057028598805513 valid acc 15/16
Epoch 14 loss 0.3443851098719574 valid acc 16/16
Epoch 14 loss 0.023599569598002357 valid acc 16/16
Epoch 14 loss 0.6526027214468119 valid acc 15/16
Epoch 14 loss 0.7195338366504689 valid acc 16/16
Epoch 14 loss 0.2578686646058382 valid acc 16/16
Epoch 14 loss 0.34332316226736825 valid acc 16/16
Epoch 14 loss 0.06590784277570438 valid acc 16/16
Epoch 14 loss 0.22332620386295793 valid acc 16/16
Epoch 14 loss 0.2863506426813343 valid acc 16/16
Epoch 14 loss 0.1229819904146901 valid acc 16/16
Epoch 14 loss 0.18196088483171102 valid acc 16/16
Epoch 14 loss 0.05261505987091039 valid acc 16/16
Epoch 14 loss 0.07671413176923525 valid acc 16/16
Epoch 14 loss 0.472397086967208 valid acc 16/16
Epoch 14 loss 0.25037512863952677 valid acc 16/16
Epoch 14 loss 0.0753553900148819 valid acc 16/16
Epoch 14 loss 0.07949600287067105 valid acc 16/16
Epoch 14 loss 0.27500845859863526 valid acc 16/16
Epoch 14 loss 0.1251663612438364 valid acc 16/16
Epoch 14 loss 0.3577487678184424 valid acc 16/16
Epoch 14 loss 0.17767846630870177 valid acc 16/16
Epoch 14 loss 0.11931771511319 valid acc 16/16
Epoch 14 loss 0.09932179548472697 valid acc 16/16
Epoch 14 loss 0.13614889865593577 valid acc 16/16
Epoch 14 loss 0.20868383025008796 valid acc 16/16
Epoch 14 loss 0.21868586299872794 valid acc 16/16
Epoch 14 loss 0.6106625303914448 valid acc 16/16
Epoch 14 loss 0.029889778154537305 valid acc 16/16
Epoch 14 loss 0.3752164821098627 valid acc 16/16
Epoch 14 loss 0.11896312043499857 valid acc 16/16
Epoch 14 loss 0.21394362152270097 valid acc 16/16
Epoch 14 loss 0.0377063428812654 valid acc 16/16
Epoch 14 loss 0.6364194099883972 valid acc 16/16
Epoch 15 loss 0.07859623702606933 valid acc 16/16
Epoch 15 loss 0.08685546320300022 valid acc 16/16
Epoch 15 loss 0.08322258265352428 valid acc 16/16
Epoch 15 loss 0.030290981523464156 valid acc 16/16
Epoch 15 loss 0.029240450761696568 valid acc 16/16
Epoch 15 loss 0.21548902413754872 valid acc 16/16
Epoch 15 loss 0.9094920208336637 valid acc 15/16
Epoch 15 loss 0.2558831457850584 valid acc 16/16
Epoch 15 loss 0.37860458415444304 valid acc 16/16
Epoch 15 loss 0.19145752327310983 valid acc 16/16
Epoch 15 loss 0.10383851944091699 valid acc 16/16
Epoch 15 loss 0.34294644803945784 valid acc 16/16
Epoch 15 loss 0.459044686722177 valid acc 16/16
Epoch 15 loss 0.09701564270505098 valid acc 16/16
Epoch 15 loss 0.6119972325301775 valid acc 16/16
Epoch 15 loss 0.22720632706999933 valid acc 16/16
Epoch 15 loss 0.3876952728001209 valid acc 16/16
Epoch 15 loss 1.0393059143542993 valid acc 16/16
Epoch 15 loss 0.7503281768634178 valid acc 16/16
Epoch 15 loss 0.5702857055953239 valid acc 16/16
Epoch 15 loss 0.8490020467614056 valid acc 16/16
Epoch 15 loss 0.16283521039513704 valid acc 16/16
Epoch 15 loss 0.2195323417275589 valid acc 16/16
Epoch 15 loss 0.11923961722846682 valid acc 16/16
Epoch 15 loss 0.07936980778678182 valid acc 16/16
Epoch 15 loss 0.26995967172561747 valid acc 16/16
Epoch 15 loss 0.1527631189535854 valid acc 16/16
Epoch 15 loss 0.06881784349942727 valid acc 16/16
Epoch 15 loss 0.6779499666395556 valid acc 16/16
Epoch 15 loss 0.032026279926912704 valid acc 16/16
Epoch 15 loss 0.06828842353495745 valid acc 16/16
Epoch 15 loss 0.10082024857645616 valid acc 16/16
Epoch 15 loss 0.15058389927154942 valid acc 16/16
Epoch 15 loss 0.25718315266159564 valid acc 16/16
Epoch 15 loss 0.2084894859835385 valid acc 15/16
Epoch 15 loss 0.04592052990285933 valid acc 16/16
Epoch 15 loss 0.42153862351171084 valid acc 15/16
Epoch 15 loss 0.12789581204987718 valid acc 15/16
Epoch 15 loss 0.1108786025597916 valid acc 15/16
Epoch 15 loss 0.05084366862450507 valid acc 16/16
Epoch 15 loss 0.2715488676401089 valid acc 15/16
Epoch 15 loss 0.08374860217387367 valid acc 16/16
Epoch 15 loss 0.22150631038824453 valid acc 16/16
Epoch 15 loss 0.1505916513451121 valid acc 16/16
Epoch 15 loss 0.2672875493832132 valid acc 16/16
Epoch 15 loss 0.08342279000662439 valid acc 16/16
Epoch 15 loss 0.07120421078917882 valid acc 15/16
Epoch 15 loss 0.2879434959108303 valid acc 16/16
Epoch 15 loss 0.12620876206289328 valid acc 16/16
Epoch 15 loss 0.12914637771290619 valid acc 16/16
Epoch 15 loss 0.3026960383062709 valid acc 16/16
Epoch 15 loss 0.2954159178254045 valid acc 16/16
Epoch 15 loss 0.013871713061672564 valid acc 16/16
Epoch 15 loss 0.048855063500454654 valid acc 16/16
Epoch 15 loss 0.17073288606331422 valid acc 16/16
Epoch 15 loss 0.11946129537152517 valid acc 16/16
Epoch 15 loss 0.21643406670308923 valid acc 16/16
Epoch 15 loss 0.14929189002018367 valid acc 16/16
Epoch 15 loss 0.3349640555550278 valid acc 16/16
Epoch 15 loss 0.13037399904556746 valid acc 16/16
Epoch 15 loss 0.24306100238465134 valid acc 16/16
Epoch 15 loss 0.348819133897682 valid acc 16/16
Epoch 15 loss 0.34325956951864584 valid acc 16/16
Epoch 16 loss 0.006562750545497709 valid acc 16/16
Epoch 16 loss 0.754702362755087 valid acc 16/16
Epoch 16 loss 0.24511948044083404 valid acc 16/16
Epoch 16 loss 0.12963947089966948 valid acc 16/16
Epoch 16 loss 0.04723729673365645 valid acc 16/16
Epoch 16 loss 0.02362414242913502 valid acc 16/16
Epoch 16 loss 0.5678551473043384 valid acc 16/16
Epoch 16 loss 0.0917075998635879 valid acc 16/16
Epoch 16 loss 0.07930723215519042 valid acc 16/16
Epoch 16 loss 0.06787664728237697 valid acc 16/16
Epoch 16 loss 0.12018051581937378 valid acc 16/16
Epoch 16 loss 0.31561634034299213 valid acc 16/16
Epoch 16 loss 0.585153175223831 valid acc 16/16
Epoch 16 loss 0.6078448766493385 valid acc 16/16
Epoch 16 loss 0.28819152664445447 valid acc 16/16
Epoch 16 loss 0.1841228907349709 valid acc 16/16
Epoch 16 loss 0.15576020802078258 valid acc 16/16
Epoch 16 loss 0.1373691921323522 valid acc 16/16
Epoch 16 loss 0.17445589360625469 valid acc 16/16
Epoch 16 loss 0.16804568533637396 valid acc 16/16
Epoch 16 loss 0.5747129953895822 valid acc 16/16
Epoch 16 loss 0.12143689711702949 valid acc 16/16
Epoch 16 loss 0.013169955625103369 valid acc 16/16
Epoch 16 loss 0.027088132125283326 valid acc 16/16
Epoch 16 loss 0.18956891033711432 valid acc 16/16
Epoch 16 loss 0.47974573144702 valid acc 16/16
Epoch 16 loss 0.041600885032214374 valid acc 16/16
Epoch 16 loss 0.09587526689358405 valid acc 16/16
Epoch 16 loss 0.14238647310153435 valid acc 16/16
Epoch 16 loss 0.012542983137902752 valid acc 16/16
Epoch 16 loss 0.06663847315743421 valid acc 16/16
Epoch 16 loss 0.10602250970510563 valid acc 16/16
Epoch 16 loss 0.08402876883643429 valid acc 16/16
Epoch 16 loss 0.08265865473992867 valid acc 16/16
Epoch 16 loss 0.0733488360965982 valid acc 16/16
Epoch 16 loss 0.35159248977539 valid acc 16/16
Epoch 16 loss 0.21612036914784327 valid acc 16/16
Epoch 16 loss 0.8763714478367421 valid acc 16/16
Epoch 16 loss 0.1125317925550452 valid acc 16/16
Epoch 16 loss 0.18503111692850718 valid acc 16/16
Epoch 16 loss 0.09964000647728521 valid acc 16/16
Epoch 16 loss 0.19243809251995542 valid acc 16/16
Epoch 16 loss 0.3803417776672317 valid acc 16/16
Epoch 16 loss 0.022171740603119494 valid acc 16/16
Epoch 16 loss 0.3215829452420144 valid acc 16/16
Epoch 16 loss 0.09882654677210784 valid acc 16/16
Epoch 16 loss 0.1944571033991826 valid acc 16/16
Epoch 16 loss 1.0673671113273144 valid acc 15/16
Epoch 16 loss 0.4460681904064621 valid acc 16/16
Epoch 16 loss 0.011274463905474597 valid acc 16/16
Epoch 16 loss 0.2418550230898947 valid acc 15/16
Epoch 16 loss 0.3665188739761035 valid acc 16/16
Epoch 16 loss 0.1911408705787005 valid acc 16/16
Epoch 16 loss 0.024024943288743883 valid acc 16/16
Epoch 16 loss 0.07768905723903002 valid acc 16/16
Epoch 16 loss 0.022930199638840798 valid acc 16/16
Epoch 16 loss 0.2496142720132712 valid acc 16/16
Epoch 16 loss 0.14526781531644795 valid acc 16/16
Epoch 16 loss 0.20442003233629885 valid acc 16/16
Epoch 16 loss 0.15766488626071234 valid acc 16/16
Epoch 16 loss 0.2471668473636469 valid acc 16/16
Epoch 16 loss 0.1032147741303669 valid acc 16/16
Epoch 16 loss 0.09659844270288231 valid acc 16/16
Epoch 17 loss 0.020772408911965967 valid acc 16/16
Epoch 17 loss 0.24407743988493544 valid acc 16/16
Epoch 17 loss 0.38282021696036106 valid acc 16/16
Epoch 17 loss 0.22794493756232048 valid acc 15/16
Epoch 17 loss 0.005958002364956028 valid acc 15/16
Epoch 17 loss 0.357256668165676 valid acc 16/16
Epoch 17 loss 0.42266908477903403 valid acc 16/16
Epoch 17 loss 0.27722377010680227 valid acc 16/16
Epoch 17 loss 0.13074625668219475 valid acc 16/16
Epoch 17 loss 0.21590170415950255 valid acc 16/16
Epoch 17 loss 0.13091893921047232 valid acc 16/16
Epoch 17 loss 0.0846094018451451 valid acc 16/16
Epoch 17 loss 0.017832099137161328 valid acc 16/16
Epoch 17 loss 0.20285799199166626 valid acc 16/16
Epoch 17 loss 0.12462162585713016 valid acc 16/16
Epoch 17 loss 0.6344067854635759 valid acc 16/16
Epoch 17 loss 0.39928773820545554 valid acc 16/16
Epoch 17 loss 0.07450964170264274 valid acc 16/16
Epoch 17 loss 0.23741953523499248 valid acc 16/16
Epoch 17 loss 0.02859093462284157 valid acc 16/16
Epoch 17 loss 0.4608333047395254 valid acc 15/16
Epoch 17 loss 0.11552167546550746 valid acc 16/16
Epoch 17 loss 0.05396009113229988 valid acc 16/16
Epoch 17 loss 0.11990112599788649 valid acc 16/16
Epoch 17 loss 0.14239440980124396 valid acc 16/16
Epoch 17 loss 0.16313667302903176 valid acc 16/16
Epoch 17 loss 0.13763157315776364 valid acc 15/16
Epoch 17 loss 0.28182101236863355 valid acc 16/16
Epoch 17 loss 0.06996129639328774 valid acc 16/16
Epoch 17 loss 0.05015513460631982 valid acc 16/16
Epoch 17 loss 0.04781693314575913 valid acc 16/16
Epoch 17 loss 0.10816873755385797 valid acc 15/16
Epoch 17 loss 0.21576534902926164 valid acc 16/16
Epoch 17 loss 0.07583783315156395 valid acc 16/16
Epoch 17 loss 0.5345209595182314 valid acc 15/16
Epoch 17 loss 0.1493213826807701 valid acc 16/16
Epoch 17 loss 0.04001963023125982 valid acc 16/16
Epoch 17 loss 0.05562461954364545 valid acc 16/16
Epoch 17 loss 0.17033013813856535 valid acc 16/16
Epoch 17 loss 0.092847519612868 valid acc 16/16
Epoch 17 loss 0.018583534070724217 valid acc 16/16
Epoch 17 loss 0.017205627569586324 valid acc 16/16
Epoch 17 loss 0.030962025595649367 valid acc 16/16
Epoch 17 loss 0.019141688876673058 valid acc 16/16
Epoch 17 loss 0.46936926506949184 valid acc 16/16
Epoch 17 loss 0.20304833116627885 valid acc 16/16
Epoch 17 loss 0.520965291103243 valid acc 16/16
Epoch 17 loss 1.0262500532246919 valid acc 16/16
Epoch 17 loss 0.2174428214565895 valid acc 16/16
Epoch 17 loss 0.11048689096219033 valid acc 16/16
Epoch 17 loss 0.42598731082390195 valid acc 16/16
Epoch 17 loss 0.3242075603424315 valid acc 16/16
Epoch 17 loss 0.5311628618104839 valid acc 15/16
Epoch 17 loss 0.28113776432753335 valid acc 16/16
Epoch 17 loss 0.15661403657505074 valid acc 16/16
Epoch 17 loss 0.022307397964845155 valid acc 16/16
Epoch 17 loss 0.2668857318328021 valid acc 15/16
Epoch 17 loss 0.14726525209348085 valid acc 16/16
Epoch 17 loss 0.060326182118687344 valid acc 16/16
Epoch 17 loss 0.08711564873620797 valid acc 16/16
Epoch 17 loss 0.33922956878392346 valid acc 16/16
Epoch 17 loss 0.08997340964680456 valid acc 16/16
Epoch 17 loss 0.26786371556056543 valid acc 16/16
Epoch 18 loss 0.002690534024975877 valid acc 16/16
Epoch 18 loss 0.18834502434767814 valid acc 16/16
Epoch 18 loss 0.042318387081890896 valid acc 16/16
Epoch 18 loss 0.5718661371947598 valid acc 16/16
Epoch 18 loss 0.2019342301415426 valid acc 16/16
Epoch 18 loss 0.038451735664772335 valid acc 16/16
Epoch 18 loss 0.053944106125310254 valid acc 16/16
Epoch 18 loss 0.07164357825717238 valid acc 16/16
Epoch 18 loss 0.0036004315060070846 valid acc 16/16
Epoch 18 loss 0.03989888340338432 valid acc 16/16
Epoch 18 loss 0.07464222199800119 valid acc 16/16
Epoch 18 loss 0.26470315068612527 valid acc 16/16
Epoch 18 loss 0.34384107403513325 valid acc 16/16
Epoch 18 loss 0.08701819505601649 valid acc 16/16
Epoch 18 loss 0.46596618995279304 valid acc 16/16
Epoch 18 loss 0.1087506113954878 valid acc 16/16
Epoch 18 loss 0.1199088819645264 valid acc 16/16
Epoch 18 loss 0.09403714156705478 valid acc 16/16
Epoch 18 loss 0.12423576804497938 valid acc 16/16
Epoch 18 loss 0.03363502367823029 valid acc 16/16
Epoch 18 loss 0.25679885236821065 valid acc 15/16
Epoch 18 loss 0.026178321227653234 valid acc 15/16
Epoch 18 loss 0.009174045630231054 valid acc 15/16
Epoch 18 loss 0.2736814083682935 valid acc 15/16
Epoch 18 loss 0.05419441414139875 valid acc 15/16
Epoch 18 loss 0.8041490117349688 valid acc 14/16
Epoch 18 loss 0.08932388194755508 valid acc 16/16
Epoch 18 loss 0.007304119994865799 valid acc 16/16
Epoch 18 loss 0.11176372013619601 valid acc 14/16
Epoch 18 loss 0.043081480979984654 valid acc 14/16
Epoch 18 loss 0.21564009686098767 valid acc 16/16
Epoch 18 loss 0.38972759689003805 valid acc 12/16
Epoch 18 loss 0.34775195252366375 valid acc 16/16
Epoch 18 loss 0.05133819077014207 valid acc 16/16
Epoch 18 loss 0.32648862243475396 valid acc 16/16
Epoch 18 loss 0.1908797941442794 valid acc 16/16
Epoch 18 loss 0.13868782225444204 valid acc 16/16
Epoch 18 loss 0.07677943519215646 valid acc 16/16
Epoch 18 loss 0.019325308934095342 valid acc 16/16
Epoch 18 loss 0.19768015643478132 valid acc 16/16
Epoch 18 loss 0.05373296678236961 valid acc 16/16
Epoch 18 loss 0.4318942332839228 valid acc 16/16
Epoch 18 loss 0.0011345903415328484 valid acc 16/16
Epoch 18 loss 0.03825831070037289 valid acc 16/16
Epoch 18 loss 0.3722123513809395 valid acc 16/16
Epoch 18 loss 0.16178481894807617 valid acc 16/16
Epoch 18 loss 0.22880876660394112 valid acc 15/16
Epoch 18 loss 0.10816666733667957 valid acc 16/16
Epoch 18 loss 0.004970508016806652 valid acc 16/16
Epoch 18 loss 0.14951174553936897 valid acc 16/16
Epoch 18 loss 0.3002699044717789 valid acc 16/16
Epoch 18 loss 0.14610416715446511 valid acc 16/16
Epoch 18 loss 0.21819886398009558 valid acc 16/16
Epoch 18 loss 0.04393475204608854 valid acc 16/16
Epoch 18 loss 0.3572481441784499 valid acc 16/16
Epoch 18 loss 0.47068416945498454 valid acc 16/16
Epoch 18 loss 0.8149944830582189 valid acc 16/16
Epoch 18 loss 0.20610776479076598 valid acc 16/16
Epoch 18 loss 0.4494042755534183 valid acc 16/16
Epoch 18 loss 0.06315008492885976 valid acc 16/16
Epoch 18 loss 0.2248336108311999 valid acc 16/16
Epoch 18 loss 0.17470386139081667 valid acc 16/16
Epoch 18 loss 1.164406198516868 valid acc 16/16
Epoch 19 loss 0.014261799081423998 valid acc 16/16
Epoch 19 loss 0.3167976136124082 valid acc 16/16
Epoch 19 loss 0.43168782689088114 valid acc 16/16
Epoch 19 loss 0.3255972703644916 valid acc 16/16
Epoch 19 loss 0.9918848955868068 valid acc 16/16
Epoch 19 loss 0.08479376623824797 valid acc 16/16
Epoch 19 loss 0.5015086699436242 valid acc 15/16
Epoch 19 loss 0.9498883662854527 valid acc 16/16
Epoch 19 loss 0.9144346417187716 valid acc 15/16
Epoch 19 loss 0.36806154395251517 valid acc 16/16
Epoch 19 loss 0.27833696754663806 valid acc 16/16
Epoch 19 loss 0.11016314738356286 valid acc 16/16
Epoch 19 loss 0.3179635754400041 valid acc 16/16
Epoch 19 loss 0.2933627007871484 valid acc 16/16
Epoch 19 loss 0.13317113024465643 valid acc 16/16
Epoch 19 loss 0.37991583828253833 valid acc 16/16
Epoch 19 loss 0.34634269834144393 valid acc 16/16
Epoch 19 loss 0.13429095298430982 valid acc 16/16
Epoch 19 loss 0.12285643305830374 valid acc 16/16
Epoch 19 loss 0.29298046888583973 valid acc 16/16
Epoch 19 loss 0.11318004184422903 valid acc 16/16
Epoch 19 loss 0.2986121883690143 valid acc 16/16
Epoch 19 loss 0.11630664777158695 valid acc 16/16
Epoch 19 loss 0.033216695647025896 valid acc 16/16
Epoch 19 loss 0.3027935556990592 valid acc 16/16
Epoch 19 loss 0.06973889693679391 valid acc 16/16
Epoch 19 loss 0.2660834376791699 valid acc 16/16
Epoch 19 loss 0.457717979479376 valid acc 16/16
Epoch 19 loss 0.29404989868763987 valid acc 16/16
Epoch 19 loss 0.19419192281656628 valid acc 16/16
Epoch 19 loss 0.05831381579176437 valid acc 16/16
Epoch 19 loss 0.18317069015285875 valid acc 16/16
Epoch 19 loss 0.050934168339042185 valid acc 16/16
Epoch 19 loss 0.3767931531771281 valid acc 16/16
Epoch 19 loss 0.1497149166402319 valid acc 16/16
Epoch 19 loss 0.13748835860537728 valid acc 16/16
Epoch 19 loss 0.05814510003135431 valid acc 16/16
Epoch 19 loss 0.18371082840464867 valid acc 15/16
Epoch 19 loss 0.04728871251673744 valid acc 15/16
Epoch 19 loss 0.4029429811914105 valid acc 15/16
Epoch 19 loss 0.6487251010865562 valid acc 16/16
Epoch 19 loss 0.05570071602863752 valid acc 16/16
Epoch 19 loss 0.02488730221289065 valid acc 16/16
Epoch 19 loss 0.030342291738742033 valid acc 16/16
Epoch 19 loss 0.34504192969440456 valid acc 16/16
Epoch 19 loss 0.01189042202726237 valid acc 16/16
Epoch 19 loss 0.04293676857520806 valid acc 16/16
Epoch 19 loss 0.35645472573089587 valid acc 16/16
Epoch 19 loss 0.16014772290637075 valid acc 16/16
Epoch 19 loss 0.29653426353363704 valid acc 16/16
Epoch 19 loss 0.09634648574544791 valid acc 16/16
Epoch 19 loss 0.1426342229346218 valid acc 16/16
Epoch 19 loss 0.0253350743599301 valid acc 16/16
Epoch 19 loss 0.012043859739926832 valid acc 16/16
Epoch 19 loss 0.23820465540949173 valid acc 16/16
Epoch 19 loss 0.11842159061871754 valid acc 16/16
Epoch 19 loss 0.03210552142972195 valid acc 16/16
Epoch 19 loss 0.004596316745403284 valid acc 16/16
Epoch 19 loss 0.47609124091984 valid acc 14/16
Epoch 19 loss 1.595549733835131 valid acc 16/16
Epoch 19 loss 0.1127139187620263 valid acc 16/16
Epoch 19 loss 0.08750694463699016 valid acc 16/16
Epoch 19 loss 0.3519900532816613 valid acc 16/16
Epoch 20 loss 0.005821668460745055 valid acc 16/16
Epoch 20 loss 0.14604104314291033 valid acc 16/16
Epoch 20 loss 0.18233178442858905 valid acc 16/16
Epoch 20 loss 0.2078720053223631 valid acc 16/16
Epoch 20 loss 0.010635537498940528 valid acc 16/16
Epoch 20 loss 0.1798511366074504 valid acc 16/16
Epoch 20 loss 0.40794861746677097 valid acc 16/16
Epoch 20 loss 0.06335037878879901 valid acc 16/16
Epoch 20 loss 0.5573895590151661 valid acc 16/16
Epoch 20 loss 0.06890284213599795 valid acc 16/16
Epoch 20 loss 0.4806484361850976 valid acc 16/16
Epoch 20 loss 0.24928540957284234 valid acc 16/16
Epoch 20 loss 0.6821917522181858 valid acc 16/16
Epoch 20 loss 0.16215923639003937 valid acc 16/16
Epoch 20 loss 0.5082419491017817 valid acc 16/16
Epoch 20 loss 0.017889761247548264 valid acc 16/16
Epoch 20 loss 0.16762277035247175 valid acc 16/16
Epoch 20 loss 0.2785292933535607 valid acc 16/16
Epoch 20 loss 0.40448849803675163 valid acc 16/16
Epoch 20 loss 0.24966469982581413 valid acc 16/16
Epoch 20 loss 0.0405353943225889 valid acc 16/16
Epoch 20 loss 0.03422086104582667 valid acc 16/16
Epoch 20 loss 0.1424767600128849 valid acc 16/16
Epoch 20 loss 0.11338044927051483 valid acc 16/16
Epoch 20 loss 0.22741890463827263 valid acc 16/16
Epoch 20 loss 0.4242798477777039 valid acc 16/16
Epoch 20 loss 0.05732137634210388 valid acc 16/16
Epoch 20 loss 0.167273403368967 valid acc 16/16
Epoch 20 loss 0.08518407982141074 valid acc 16/16
Epoch 20 loss 0.21197184952041365 valid acc 16/16
Epoch 20 loss 0.36430148794831924 valid acc 16/16
Epoch 20 loss 0.268402023932567 valid acc 16/16
Epoch 20 loss 0.2151393550938997 valid acc 16/16
Epoch 20 loss 0.02883845098098381 valid acc 16/16
Epoch 20 loss 0.4064480553585955 valid acc 16/16
Epoch 20 loss 0.13662153212185846 valid acc 16/16
Epoch 20 loss 0.21813097722049793 valid acc 15/16
Epoch 20 loss 0.6840712167695638 valid acc 16/16
Epoch 20 loss 0.205441091282154 valid acc 16/16
Epoch 20 loss 0.1483398186089343 valid acc 16/16
Epoch 20 loss 0.14192380416565398 valid acc 16/16
Epoch 20 loss 0.04318003907834783 valid acc 16/16
Epoch 20 loss 0.015501403434503014 valid acc 16/16
Epoch 20 loss 0.07573016708617708 valid acc 16/16
Epoch 20 loss 0.13262815371643105 valid acc 16/16
Epoch 20 loss 0.031028153716602613 valid acc 16/16
Epoch 20 loss 0.6786997216655299 valid acc 16/16
Epoch 20 loss 0.3867106405022398 valid acc 16/16
Epoch 20 loss 0.06567632991354677 valid acc 16/16
Epoch 20 loss 0.020908111195773404 valid acc 16/16
Epoch 20 loss 0.056161769287952135 valid acc 16/16
Epoch 20 loss 0.09295574995509256 valid acc 16/16
Epoch 20 loss 0.04655106375177104 valid acc 16/16
Epoch 20 loss 0.022231291395597264 valid acc 16/16
Epoch 20 loss 0.09352012584393254 valid acc 16/16
Epoch 20 loss 0.010019964020649086 valid acc 16/16
Epoch 20 loss 0.2729373598709776 valid acc 16/16
Epoch 20 loss 0.016930765747076193 valid acc 16/16
Epoch 20 loss 0.11409107391993178 valid acc 16/16
Epoch 20 loss 0.3668282629597431 valid acc 16/16
Epoch 20 loss 0.09492712566289507 valid acc 16/16
Epoch 20 loss 0.1752326174929441 valid acc 16/16
Epoch 20 loss 0.20274479998076067 valid acc 16/16
Epoch 21 loss 0.008549337637533128 valid acc 16/16
Epoch 21 loss 0.35901368768793535 valid acc 16/16
Epoch 21 loss 0.30931032853329193 valid acc 16/16
Epoch 21 loss 0.1865650898112082 valid acc 16/16
Epoch 21 loss 0.019333102745425768 valid acc 16/16
Epoch 21 loss 0.05311655368166385 valid acc 16/16
Epoch 21 loss 0.34384988334708255 valid acc 16/16
Epoch 21 loss 0.1240351795287934 valid acc 16/16
Epoch 21 loss 1.1352163494538372 valid acc 16/16
Epoch 21 loss 0.01895269693813867 valid acc 16/16
Epoch 21 loss 0.5178885353665987 valid acc 16/16
Epoch 21 loss 0.32917036073734834 valid acc 16/16
Epoch 21 loss 0.019850089378871894 valid acc 16/16
Epoch 21 loss 0.2649334216649408 valid acc 16/16
Epoch 21 loss 0.3017847557957714 valid acc 16/16
Epoch 21 loss 0.4199486947552711 valid acc 16/16
Epoch 21 loss 0.7793133548405539 valid acc 16/16
Epoch 21 loss 0.09575160099132579 valid acc 16/16
Epoch 21 loss 0.03139603989650708 valid acc 16/16
Epoch 21 loss 0.23997757064563122 valid acc 16/16
Epoch 21 loss 0.06996955591865395 valid acc 16/16
Epoch 21 loss 0.45809091207249597 valid acc 16/16
Epoch 21 loss 0.266435817260351 valid acc 14/16
Epoch 21 loss 1.038256458517426 valid acc 16/16
Epoch 21 loss 0.7144130932588113 valid acc 16/16
Epoch 21 loss 0.26531636351744814 valid acc 16/16
Epoch 21 loss 0.21203669061341102 valid acc 16/16
Epoch 21 loss 0.31883251125988477 valid acc 16/16
Epoch 21 loss 0.04079172160297262 valid acc 15/16
Epoch 21 loss 0.016007378572499065 valid acc 15/16
Epoch 21 loss 0.13958086242231238 valid acc 16/16
Epoch 21 loss 0.32745907040496275 valid acc 16/16
Epoch 21 loss 0.3275364121683119 valid acc 16/16
Epoch 21 loss 0.02747566518594835 valid acc 16/16
Epoch 21 loss 0.18840848345718297 valid acc 16/16
Epoch 21 loss 0.2062202539107159 valid acc 16/16
Epoch 21 loss 0.17956673851950727 valid acc 16/16
Epoch 21 loss 0.027250071462449454 valid acc 16/16
Epoch 21 loss 0.17682753329029477 valid acc 16/16
Epoch 21 loss 0.08756894336655358 valid acc 16/16
Epoch 21 loss 0.20070967143833296 valid acc 16/16
Epoch 21 loss 0.26581221019737833 valid acc 16/16
Epoch 21 loss 0.2391341888830234 valid acc 16/16
Epoch 21 loss 0.07378154568340467 valid acc 16/16
Epoch 21 loss 0.4735223145060583 valid acc 16/16
Epoch 21 loss 0.028207089508259675 valid acc 16/16
Epoch 21 loss 0.3391715843270071 valid acc 16/16
Epoch 21 loss 0.08344229155318855 valid acc 16/16
Epoch 21 loss 0.08421060310358397 valid acc 16/16
Epoch 21 loss 0.1749964676462626 valid acc 16/16
Epoch 21 loss 0.07474080020799134 valid acc 16/16
Epoch 21 loss 0.04141658522125802 valid acc 16/16
Epoch 21 loss 0.20761344225260914 valid acc 16/16
Epoch 21 loss 0.011198693137915994 valid acc 16/16
Epoch 21 loss 0.0874987925892971 valid acc 16/16
Epoch 21 loss 0.02564126477793799 valid acc 16/16
Epoch 21 loss 0.07330305038017482 valid acc 16/16
Epoch 21 loss 0.12389409340817448 valid acc 16/16
Epoch 21 loss 0.516388717001073 valid acc 16/16
Epoch 21 loss 0.2866270497404885 valid acc 16/16
Epoch 21 loss 0.1150511445432259 valid acc 16/16
Epoch 21 loss 0.11244885328213328 valid acc 16/16
Epoch 21 loss 0.4539347718855861 valid acc 16/16
Epoch 22 loss 0.006258121970427966 valid acc 16/16
Epoch 22 loss 0.1904267124001754 valid acc 16/16
Epoch 22 loss 0.15310273874121677 valid acc 16/16
Epoch 22 loss 0.332825868048409 valid acc 16/16
Epoch 22 loss 0.049500898391599935 valid acc 16/16
Epoch 22 loss 0.03764568012387831 valid acc 16/16
Epoch 22 loss 0.1948596320611085 valid acc 16/16
Epoch 22 loss 0.21479209611392705 valid acc 16/16
Epoch 22 loss 0.17822110309560862 valid acc 16/16
Epoch 22 loss 0.08072077281988976 valid acc 16/16
Epoch 22 loss 0.14879598306635428 valid acc 16/16
Epoch 22 loss 0.5989677614010392 valid acc 16/16
Epoch 22 loss 0.17148576707577068 valid acc 16/16
Epoch 22 loss 0.10053840875522058 valid acc 16/16
Epoch 22 loss 0.2586320167321502 valid acc 16/16
Epoch 22 loss 0.061570225959642844 valid acc 16/16
Epoch 22 loss 1.3111500876848061 valid acc 16/16
Epoch 22 loss 0.2492304417919313 valid acc 16/16
Epoch 22 loss 0.1532530131044037 valid acc 16/16
Epoch 22 loss 0.19025457456220815 valid acc 16/16
Epoch 22 loss 0.1697486712031453 valid acc 16/16
Epoch 22 loss 0.23492885232754213 valid acc 16/16
Epoch 22 loss 0.0456674206634301 valid acc 16/16
Epoch 22 loss 0.24006699037636345 valid acc 16/16
Epoch 22 loss 0.636490858750963 valid acc 16/16
Epoch 22 loss 0.2661514027595008 valid acc 16/16
Epoch 22 loss 0.026994512815570415 valid acc 16/16
Epoch 22 loss 0.31336391958164755 valid acc 16/16
Epoch 22 loss 0.23168842508988144 valid acc 16/16
Epoch 22 loss 0.09608335696003723 valid acc 16/16
Epoch 22 loss 0.08184777639023211 valid acc 16/16
Epoch 22 loss 0.23683232709423674 valid acc 16/16
Epoch 22 loss 0.08768521458848444 valid acc 16/16
Epoch 22 loss 0.043389903394820406 valid acc 16/16
Epoch 22 loss 0.0585783494831687 valid acc 16/16
Epoch 22 loss 0.09637844152285596 valid acc 16/16
Epoch 22 loss 0.0962649663690047 valid acc 16/16
Epoch 22 loss 0.045933591242323546 valid acc 16/16
Epoch 22 loss 0.0581176794618769 valid acc 16/16
Epoch 22 loss 0.047476476207595014 valid acc 16/16
Epoch 22 loss 0.0348710321718973 valid acc 16/16
Epoch 22 loss 0.20721200153879715 valid acc 16/16
Epoch 22 loss 0.23811892057061274 valid acc 16/16
Epoch 22 loss 0.08496407516097204 valid acc 16/16
Epoch 22 loss 0.3198167395908533 valid acc 15/16
Epoch 22 loss 0.0628651231201669 valid acc 16/16
Epoch 22 loss 0.11086852091697846 valid acc 16/16
Epoch 22 loss 0.24555989490755564 valid acc 16/16
Epoch 22 loss 0.007665930696290291 valid acc 16/16
Epoch 22 loss 0.0028997917732328515 valid acc 16/16
Epoch 22 loss 0.01584212804995283 valid acc 16/16
Epoch 22 loss 0.048037409371156914 valid acc 16/16
Epoch 22 loss 0.373090764137763 valid acc 16/16
Epoch 22 loss 0.16146868992716057 valid acc 16/16
Epoch 22 loss 0.19421344699305088 valid acc 16/16
Epoch 22 loss 0.03768066949198834 valid acc 16/16
Epoch 22 loss 0.5505042707833404 valid acc 16/16
Epoch 22 loss 0.29747135266376284 valid acc 16/16
Epoch 22 loss 0.29871250600290306 valid acc 16/16
Epoch 22 loss 0.18582922209528135 valid acc 16/16
Epoch 22 loss 0.03397405935350528 valid acc 16/16
Epoch 22 loss 0.28608117812024153 valid acc 16/16
Epoch 22 loss 0.20657642750684707 valid acc 16/16
Epoch 23 loss 0.00025652094404304826 valid acc 16/16
Epoch 23 loss 0.6277184704733484 valid acc 16/16
Epoch 23 loss 0.030198166857933084 valid acc 16/16
Epoch 23 loss 0.35575165032806905 valid acc 16/16
Epoch 23 loss 0.08084675531193286 valid acc 16/16
Epoch 23 loss 0.15889131139296353 valid acc 16/16
Epoch 23 loss 0.3313748131078858 valid acc 16/16
Epoch 23 loss 0.14738005551329236 valid acc 16/16
Epoch 23 loss 0.14178441142592918 valid acc 16/16
Epoch 23 loss 0.27088990933164936 valid acc 16/16
Epoch 23 loss 0.04380974813200422 valid acc 16/16
Epoch 23 loss 0.6379152541740332 valid acc 15/16
Epoch 23 loss 0.5714199313057233 valid acc 16/16
Epoch 23 loss 0.31332776001478835 valid acc 16/16
Epoch 23 loss 0.07953755504755539 valid acc 16/16
Epoch 23 loss 0.1579197925040453 valid acc 16/16
Epoch 23 loss 0.3112463614246414 valid acc 16/16
Epoch 23 loss 0.0670512906277621 valid acc 16/16
Epoch 23 loss 0.06519625681794261 valid acc 16/16
Epoch 23 loss 0.12662831628592658 valid acc 16/16
Epoch 23 loss 0.296794687268781 valid acc 15/16
Epoch 23 loss 0.07386628434458697 valid acc 15/16
Epoch 23 loss 0.043272018392707334 valid acc 16/16
Epoch 23 loss 0.16934965036851513 valid acc 16/16
Epoch 23 loss 0.028046583007094172 valid acc 16/16
Epoch 23 loss 0.005536533929044407 valid acc 16/16
Epoch 23 loss 0.13294926094949727 valid acc 16/16
Epoch 23 loss 0.00887817561129245 valid acc 16/16
Epoch 23 loss 0.04051438824250908 valid acc 16/16
Epoch 23 loss 0.3340210137575533 valid acc 16/16
Epoch 23 loss 0.01607301348580714 valid acc 16/16
Epoch 23 loss 0.057949195170883094 valid acc 16/16
Epoch 23 loss 0.08997957268350121 valid acc 16/16
Epoch 23 loss 0.25920082287654694 valid acc 16/16
Epoch 23 loss 0.13209974923062817 valid acc 16/16
Epoch 23 loss 0.3839761021227217 valid acc 16/16
Epoch 23 loss 0.2908516616333105 valid acc 16/16
Epoch 23 loss 0.11496488141099592 valid acc 16/16
Epoch 23 loss 0.18689549998450627 valid acc 16/16
Epoch 23 loss 0.21470302350828252 valid acc 16/16
Epoch 23 loss 0.44516314237055715 valid acc 16/16
Epoch 23 loss 0.31419657953992775 valid acc 16/16
Epoch 23 loss 0.03944397161257729 valid acc 16/16
Epoch 23 loss 0.10724440166259488 valid acc 16/16
Epoch 23 loss 1.141758261962409 valid acc 16/16
Epoch 23 loss 0.6491493747865851 valid acc 16/16
Epoch 23 loss 0.32334868348973583 valid acc 16/16
Epoch 23 loss 0.18389172795443093 valid acc 16/16
Epoch 23 loss 0.3332301993789057 valid acc 16/16
Epoch 23 loss 0.05269427698289242 valid acc 16/16
Epoch 23 loss 0.10636565857453245 valid acc 16/16
Epoch 23 loss 0.4255917660029916 valid acc 16/16
Epoch 23 loss 0.2770564917866077 valid acc 16/16
Epoch 23 loss 0.2530648948335714 valid acc 16/16
Epoch 23 loss 0.08477471729372965 valid acc 16/16
Epoch 23 loss 0.1586898829436858 valid acc 16/16
Epoch 23 loss 0.12327440095756415 valid acc 16/16
Epoch 23 loss 0.02670793033224278 valid acc 16/16
Epoch 23 loss 0.22560741658244865 valid acc 16/16
Epoch 23 loss 0.05714801129415842 valid acc 16/16
Epoch 23 loss 0.03068486159265016 valid acc 16/16
Epoch 23 loss 0.3445038271000941 valid acc 16/16
Epoch 23 loss 0.07220019782400278 valid acc 16/16
Epoch 24 loss 9.156567464119192e-05 valid acc 16/16
Epoch 24 loss 0.11331685232412231 valid acc 16/16
Epoch 24 loss 0.09335664389139342 valid acc 16/16
Epoch 24 loss 0.06564715331603824 valid acc 16/16
Epoch 24 loss 0.041546939022022555 valid acc 16/16
Epoch 24 loss 0.01781524021016523 valid acc 16/16
Epoch 24 loss 0.09775065519892212 valid acc 16/16
Epoch 24 loss 0.11197651086267157 valid acc 16/16
Epoch 24 loss 0.11669166877818551 valid acc 16/16
Epoch 24 loss 0.04276644041034916 valid acc 16/16
Epoch 24 loss 0.1341421249419505 valid acc 16/16
Epoch 24 loss 0.16574132141501147 valid acc 16/16
Epoch 24 loss 0.01225011259410469 valid acc 16/16
Epoch 24 loss 0.1818585618819969 valid acc 16/16
Epoch 24 loss 0.04786386790314734 valid acc 16/16
Epoch 24 loss 0.3194833609312564 valid acc 16/16
Epoch 24 loss 0.09949902042847991 valid acc 16/16
Epoch 24 loss 0.008081448338427943 valid acc 16/16
Epoch 24 loss 0.019431258394394912 valid acc 16/16
Epoch 24 loss 0.11384099313238372 valid acc 16/16
Epoch 24 loss 0.3552918643088736 valid acc 15/16
Epoch 24 loss 0.1660353272416003 valid acc 15/16
Epoch 24 loss 0.44561778067805347 valid acc 15/16
Epoch 24 loss 0.471761444022028 valid acc 14/16
Epoch 24 loss 1.2598176286950988 valid acc 15/16
Epoch 24 loss 0.5449900714612399 valid acc 16/16
Epoch 24 loss 0.349515431454362 valid acc 16/16
Epoch 24 loss 0.20543364936427255 valid acc 16/16
Epoch 24 loss 0.23622591523950007 valid acc 16/16
Epoch 24 loss 0.082619771529447 valid acc 16/16
Epoch 24 loss 0.3945577937851621 valid acc 16/16
Epoch 24 loss 0.0671484639816472 valid acc 16/16
Epoch 24 loss 0.17552021435719023 valid acc 16/16
Epoch 24 loss 0.04508469155823969 valid acc 16/16
Epoch 24 loss 0.08447478018532206 valid acc 16/16
Epoch 24 loss 0.029429399037146586 valid acc 16/16
Epoch 24 loss 0.5894416792119102 valid acc 16/16
Epoch 24 loss 0.12063911593659571 valid acc 16/16
Epoch 24 loss 0.04750693567688846 valid acc 16/16
Epoch 24 loss 0.025533278324233688 valid acc 16/16
Epoch 24 loss 0.08893180572770704 valid acc 16/16
Epoch 24 loss 0.10623811356548418 valid acc 15/16
Epoch 24 loss 0.07657391171213634 valid acc 16/16
Epoch 24 loss 0.005905200216695938 valid acc 16/16
Epoch 24 loss 0.04270834532076695 valid acc 16/16
Epoch 24 loss 0.11693687631525862 valid acc 16/16
Epoch 24 loss 0.0030978611999014083 valid acc 16/16
Epoch 24 loss 0.05212183846283508 valid acc 16/16
Epoch 24 loss 0.6862749990552278 valid acc 16/16
Epoch 24 loss 0.028926900602036953 valid acc 16/16
Epoch 24 loss 0.09471141028430796 valid acc 16/16
Epoch 24 loss 0.11019880170201984 valid acc 16/16
Epoch 24 loss 0.08513899663979202 valid acc 16/16
Epoch 24 loss 0.14418300333168574 valid acc 16/16
Epoch 24 loss 0.031593019501173714 valid acc 16/16
Epoch 24 loss 0.09361616268236067 valid acc 16/16
Epoch 24 loss 0.32048324767252734 valid acc 16/16
Epoch 24 loss 0.007891690322952116 valid acc 16/16
Epoch 24 loss 0.34852009534179146 valid acc 16/16
Epoch 24 loss 0.07133425405582278 valid acc 16/16
Epoch 24 loss 0.3117891794730893 valid acc 16/16
Epoch 24 loss 0.35508155955977644 valid acc 16/16
Epoch 24 loss 0.04947247137160321 valid acc 16/16
Epoch 25 loss 0.023624490344091975 valid acc 16/16
Epoch 25 loss 0.18440423981553228 valid acc 15/16
Epoch 25 loss 0.7504961952349352 valid acc 16/16
Epoch 25 loss 0.2634386265039287 valid acc 16/16
Epoch 25 loss 0.012915281740697004 valid acc 16/16
Epoch 25 loss 0.12854365853375216 valid acc 16/16
Epoch 25 loss 0.7301297451197024 valid acc 16/16
Epoch 25 loss 0.3332636158476618 valid acc 15/16
Epoch 25 loss 0.09637654304666798 valid acc 15/16
Epoch 25 loss 0.021373806275345797 valid acc 15/16
Epoch 25 loss 0.10075707031045478 valid acc 15/16
Epoch 25 loss 0.15131294583217608 valid acc 15/16
Epoch 25 loss 0.04365472893213118 valid acc 15/16
Epoch 25 loss 0.32128754263198916 valid acc 15/16
Epoch 25 loss 0.09626583872843966 valid acc 15/16
Epoch 25 loss 0.03563170374246938 valid acc 15/16
Epoch 25 loss 0.10680553814145755 valid acc 15/16
Epoch 25 loss 0.9921697065603268 valid acc 16/16
Epoch 25 loss 0.14131666081713146 valid acc 16/16
Epoch 25 loss 0.039913520325231266 valid acc 16/16
Epoch 25 loss 0.15161653815763265 valid acc 16/16
Epoch 25 loss 0.07710981695874108 valid acc 16/16
Epoch 25 loss 0.0028785352786908636 valid acc 16/16
Epoch 25 loss 0.008944118198336237 valid acc 16/16
Epoch 25 loss 0.1658282787462238 valid acc 16/16
Epoch 25 loss 0.13464606716203148 valid acc 16/16
Epoch 25 loss 0.25347150428706317 valid acc 14/16
Epoch 25 loss 0.19843030565744324 valid acc 15/16
Epoch 25 loss 0.05554900133799251 valid acc 15/16
Epoch 25 loss 0.0871412369086193 valid acc 15/16
Epoch 25 loss 0.3426980621240467 valid acc 15/16
Epoch 25 loss 0.13212802304694898 valid acc 15/16
Epoch 25 loss 0.020711983253675348 valid acc 15/16
Epoch 25 loss 0.1809380322049433 valid acc 15/16
Epoch 25 loss 0.42914798600940685 valid acc 16/16
Epoch 25 loss 0.48680609940528 valid acc 15/16
Epoch 25 loss 0.09396449375392574 valid acc 16/16
Epoch 25 loss 0.4992720728615288 valid acc 16/16
Epoch 25 loss 0.08522660991191894 valid acc 16/16
Epoch 25 loss 0.03977456398990448 valid acc 16/16
Epoch 25 loss 0.01999835117642368 valid acc 16/16
Epoch 25 loss 0.013865940984324032 valid acc 16/16
Epoch 25 loss 0.08252840929080196 valid acc 16/16
Epoch 25 loss 0.06310687986770536 valid acc 16/16
Epoch 25 loss 0.11751506392193592 valid acc 16/16
Epoch 25 loss 0.009218316594238107 valid acc 16/16
Epoch 25 loss 0.0641679573021926 valid acc 16/16
Epoch 25 loss 0.27805310360131863 valid acc 16/16
Epoch 25 loss 0.16306798173339454 valid acc 16/16
Epoch 25 loss 0.008395496447333395 valid acc 16/16
Epoch 25 loss 0.05095583075825161 valid acc 16/16
Epoch 25 loss 0.16766186935728597 valid acc 16/16
Epoch 25 loss 0.19259513351152835 valid acc 15/16
Epoch 25 loss 0.04877498182664175 valid acc 16/16
Epoch 25 loss 0.26333089048797703 valid acc 15/16
Epoch 25 loss 0.019957829414662287 valid acc 15/16
Epoch 25 loss 0.5433241520705909 valid acc 15/16
Epoch 25 loss 0.031108749565910754 valid acc 15/16
Epoch 25 loss 0.0770303427616792 valid acc 15/16
Epoch 25 loss 0.06768176572204576 valid acc 15/16
Epoch 25 loss 0.06790405869066823 valid acc 16/16
Epoch 25 loss 0.1558752804733296 valid acc 16/16
Epoch 25 loss 0.17526886187271346 valid acc 16/16
Epoch 26 loss 0.09194030408509823 valid acc 16/16
Epoch 26 loss 0.22405910919900995 valid acc 16/16
Epoch 26 loss 0.034954385833359636 valid acc 16/16
Epoch 26 loss 0.14168456138787033 valid acc 16/16
Epoch 26 loss 0.04053361596125615 valid acc 16/16
Epoch 26 loss 0.037215033603246295 valid acc 16/16
Epoch 26 loss 0.20071295097544145 valid acc 16/16
Epoch 26 loss 0.18881284881917199 valid acc 16/16
Epoch 26 loss 0.053575279704427936 valid acc 15/16
Epoch 26 loss 0.005642991343055348 valid acc 15/16
Epoch 26 loss 0.01054964705096939 valid acc 15/16
Epoch 26 loss 0.07642922883945102 valid acc 16/16
Epoch 26 loss 0.026540187153392514 valid acc 16/16
Epoch 26 loss 0.013785452423085931 valid acc 16/16
Epoch 26 loss 0.20915236225086314 valid acc 16/16
Epoch 26 loss 0.09723052153736444 valid acc 16/16
Epoch 26 loss 0.1650743862570892 valid acc 16/16
Epoch 26 loss 0.016536446693890094 valid acc 16/16
Epoch 26 loss 0.09515865328186557 valid acc 16/16
Epoch 26 loss 0.020460344622376503 valid acc 16/16
Epoch 26 loss 0.016500792878549664 valid acc 16/16
Epoch 26 loss 0.024577195642484506 valid acc 16/16
Epoch 26 loss 0.03822539263402869 valid acc 16/16
Epoch 26 loss 0.11399337771406037 valid acc 16/16
Epoch 26 loss 0.08984429887442025 valid acc 16/16
Epoch 26 loss 0.32899556123575036 valid acc 16/16
Epoch 26 loss 0.30342682037365387 valid acc 16/16
Epoch 26 loss 0.17357059150415716 valid acc 16/16
Epoch 26 loss 0.2684300290266102 valid acc 16/16
Epoch 26 loss 0.0058113123921100795 valid acc 16/16
Epoch 26 loss 0.17328595764832788 valid acc 16/16
Epoch 26 loss 0.04102695733259376 valid acc 16/16
Epoch 26 loss 0.5101961016582303 valid acc 16/16
Epoch 26 loss 0.02810534827014821 valid acc 16/16
Epoch 26 loss 0.5017217765762797 valid acc 16/16
Epoch 26 loss 0.16565956657440767 valid acc 16/16
Epoch 26 loss 0.1432410103730411 valid acc 16/16
Epoch 26 loss 0.011001174668374025 valid acc 16/16
Epoch 26 loss 0.0366282719678217 valid acc 16/16
Epoch 26 loss 0.1867186086281053 valid acc 16/16
Epoch 26 loss 0.19860712707551886 valid acc 16/16
Epoch 26 loss 0.012072246483900817 valid acc 16/16
Epoch 26 loss 0.056419729747635206 valid acc 16/16
Epoch 26 loss 0.0035561578744954248 valid acc 16/16
Epoch 26 loss 0.0249805890286664 valid acc 16/16
Epoch 26 loss 0.02102092297519864 valid acc 16/16
Epoch 26 loss 0.028605090270243827 valid acc 16/16
Epoch 26 loss 0.1766216071345777 valid acc 16/16
Epoch 26 loss 0.1472402148598222 valid acc 16/16
Epoch 26 loss 0.023769668767532948 valid acc 16/16
Epoch 26 loss 0.09320186735379077 valid acc 16/16
Epoch 26 loss 0.018072482950022006 valid acc 16/16
Epoch 26 loss 0.02627142474986649 valid acc 16/16
Epoch 26 loss 0.1071164525920899 valid acc 15/16
Epoch 26 loss 0.058962087597922486 valid acc 15/16
Epoch 26 loss 0.008188826441737185 valid acc 15/16
Epoch 26 loss 0.06492830540551531 valid acc 16/16
Epoch 26 loss 0.2672146357153289 valid acc 16/16
Epoch 26 loss 0.05064815829268088 valid acc 15/16
Epoch 26 loss 0.034897035751097205 valid acc 16/16
Epoch 26 loss 0.11413007666939468 valid acc 16/16
Epoch 26 loss 0.10854148971045441 valid acc 16/16
Epoch 26 loss 0.0074994602066739136 valid acc 16/16
```