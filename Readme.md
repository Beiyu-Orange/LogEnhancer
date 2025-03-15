This document contains all the code used in the experiment, the experiment results, and the original data set.

## Code
+ tf_bert3 is the w2v-bert-bmu original model code.
+ test_gpt is the code that tests the LLMs classification.
+ SMOTE and class weight are traditional enhancement method codes.
+ The rest of the code is used for metric analysis.

## Data
+ The experimental data is stored in the B_S, LLMs and ours folders.
+ data.csv is oriiginal data set.
+ error_logs is the folder where the logs of specific build failures are stored.


| Build platform | Data set | Collected urls                                               |
| -------------- | -------- | ------------------------------------------------------------ |
| OpenSeuse      | 416      | https://build.opensuse.org/project/show/openSUSE:Factory:RISCV |
| OpenEuler      | 602      | https://build.openeuler.openatom.cn/project/show/openEuler:Mainline:RISC-V |
| Tarsier        | 39       | https://build.tarsier-infra.com/project/show/openEuler:23.03 |



