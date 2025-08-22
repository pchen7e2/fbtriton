Issue: ptxas optimizations cause a larger numerical error.

Summary: In this [triplet kernel](https://github.com/meta-pytorch/AccelKernels/blob/14222fd1c4acc8f42fa5dbca4d8af77266b7f0d7/triplet_attention/triplet/ops/tlx/fwd_ws_pingpong.py#L3), we have a layout conversion for operand A from blocked layout to dot operand layout which involves an SMEM write+read. We did a [hack](https://github.com/facebookexperimental/triton/commit/515db4b16cf7ee11457f09261a79d591dbabbd4f) to eleminate these two and yielded a great perf win but notices large numerical errors on certain shapes. Forcing ptxas opt level to 0 will resolve the NE (but perf is a lot worse).

How to reproduce:
- Download and install this [kernel](https://github.com/meta-pytorch/AccelKernels/tree/14222fd1c4acc8f42fa5dbca4d8af77266b7f0d7/triplet_attention) (checkout commit 5b8f218cb3c1cb82b8e38902501f9cc668f823a9)
- Download and install this Triton branch (based on late May upstream main branch) [link](https://github.com/facebookexperimental/triton/tree/tlx) (checkout commit 76b98297e061b158cbea6147cef1d1bd9e6d9f68)
  - To apply the hack, apply this patch: https://github.com/facebookexperimental/triton/commit/515db4b16cf7ee11457f09261a79d591dbabbd4f
- Run this [test](https://github.com/meta-pytorch/AccelKernels/blob/14222fd1c4acc8f42fa5dbca4d8af77266b7f0d7/triplet_attention/tests/test_tlx_fwd_ws_pingpong.py#L29) (`python test_tlx_fwd_ws_pingpong.py`) w/ and w/o the hack mentioned above. Test failing means NE is bad.
- To check performance, run this [script](https://github.com/meta-pytorch/AccelKernels/blob/14222fd1c4acc8f42fa5dbca4d8af77266b7f0d7/triplet_attention/triplet/ops/tlx/fwd_ws_pingpong.py#L3) (`python fwd_ws_pingpong.py`)

Some useful attachments:
- baseline.ttgir: baseline ttgir w/o the hack. The perf is not as good as w/ the hack, but no NE.
- hack_fail.ttgir: manually edited ttgir to reflect the essence of the hack. Useful to compare with baseline.ttgir. Perf is very good, but NE is large.
- hack_pass.ttgir: compared to hack_fail.ttgir, enable only one ttg.local_alloc op (SMEM write) whose output is not used. It's supposed to be a no-op on accuracy, but it actually yielded a good NE and a bad perf.

Then to minimize the PTX difference, I moved two lines in the ttgir override to be closer to wgmma op:
- hack_pass1.ttgir: moved three lines compared to hack_pass.ttgir. Good NE.
- hack_fail1.ttgir: change the operand of ttg.alloc in hack_pass1.ttgir to another value. Bad NE.
They have minimal difference and I run with these overrides and dump the ttgir and ptx too in folder `hack_pass1_dump` and `hack_fail1_dump`. In particular, the only real difference between `./hack_pass1_dump/AOXQPRQGU2KYAFKKTICYPHADZSB237FPNMMT6OCLE37FUDSF775A/_triplet_tlx_fwd_ws_kernel_pingpong_with_hack.ptx` and `./hack_fail1_dump/AOXQPRQGU2KYAFKKTICYPHADZSB237FPNMMT6OCLE37FUDSF775A/_triplet_tlx_fwd_ws_kernel_pingpong_with_hack.ptx` is a few `stmatrix` writing from different registers. I also used `nvdisasm -c` to dump their SASS and the difference seems much larger than PTX differences. (default ptxas opt level)

Then I tried forcing opt level to be 0 [link](https://github.com/facebookexperimental/triton/blob/09907b3007c08b7d9c85c1421ce29d7ef19582e9/third_party/nvidia/backend/compiler.py#L392) and NE issue is gone.
