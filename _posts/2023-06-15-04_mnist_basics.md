---
title: "Fastai와 파이토치가 만나 꽃피운 딥러닝 챕터 4"
date: 2023-06-15 00:00:00 +0900
categories:
  - fastai
toc: true  
#image : assets/image/how-to-start.jpg # Add image post (optional)
tags:
  - ML
  - MNIST
---
Source :
- [04_mnist_basics.ipynb](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
- [Fastbook Chapter 4 questionnaire solutions (wiki)](https://forums.fast.ai/t/fastbook-chapter-4-questionnaire-solutions-wiki/67253)

### Chapter 4 full code
```
#hide
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```
```
output:  

[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m719.8/719.8 kB[0m [31m49.8 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.2/7.2 MB[0m [31m73.8 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m485.6/485.6 kB[0m [31m46.7 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m83.0 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m110.5/110.5 kB[0m [31m14.3 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m212.5/212.5 kB[0m [31m28.2 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.3/134.3 kB[0m [31m16.8 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m61.7 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m236.8/236.8 kB[0m [31m29.1 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m68.9 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m61.6 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.5/114.5 kB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m268.8/268.8 kB[0m [31m29.9 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m149.6/149.6 kB[0m [31m16.5 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m60.9 MB/s[0m eta [36m0:00:00[0m
[?25hMounted at /content/gdrive
```
```
#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')
```
```
path = untar_data(URLs.MNIST_SAMPLE)
```

<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>

<div>
  <progress value='3219456' class='' max='3214948' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.14% [3219456/3214948 00:01&lt;00:00]
</div>

```
#hide
Path.BASE_PATH = path
```
```
path.ls()
```
```
output :  
[Path('valid'),Path('train'),Path('labels.csv')]
```
```
(path/'train').ls()
```
```
output:  
[Path('train/3'),Path('train/7')]
```
```
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
```
```
output :  
[Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),Path('train/3/10091.png')...]
```
```
im3_path = threes[1]
im3 = Image.open(im3_path)
im3
```

output :  
![04_mnist_basics_8_0](https://github.com/tjwodud04/blog/assets/34568203/431bee82-eeba-4935-afab-0ab790f0d433)
```
array(im3)[4:10,4:10]
```
```
output :  
array([[  0,   0,   0,   0,   0,   0],  
       [  0,   0,   0,   0,   0,  29],  
       [  0,   0,   0,  48, 166, 224],  
       [  0,  93, 244, 249, 253, 187],  
       [  0, 107, 253, 253, 230,  48],  
       [  0,   3,  20,  20,  15,   0]], dtype=uint8)
```
```
tensor(im3)[4:10,4:10]
```
```
output : 
tensor([[  0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,  29],
        [  0,   0,   0,  48, 166, 224],
        [  0,  93, 244, 249, 253, 187],
        [  0, 107, 253, 253, 230,  48],
        [  0,   3,  20,  20,  15,   0]], dtype=torch.uint8)
```        
```
#hide_output
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

<style type="text/css">
#T_2120c_row0_col0, #T_2120c_row0_col1, #T_2120c_row0_col2, #T_2120c_row0_col3, #T_2120c_row0_col4, #T_2120c_row0_col5, #T_2120c_row0_col6, #T_2120c_row0_col7, #T_2120c_row0_col8, #T_2120c_row0_col9, #T_2120c_row0_col10, #T_2120c_row0_col11, #T_2120c_row0_col12, #T_2120c_row0_col13, #T_2120c_row0_col14, #T_2120c_row0_col15, #T_2120c_row0_col16, #T_2120c_row0_col17, #T_2120c_row1_col0, #T_2120c_row1_col1, #T_2120c_row1_col2, #T_2120c_row1_col3, #T_2120c_row1_col4, #T_2120c_row1_col15, #T_2120c_row1_col16, #T_2120c_row1_col17, #T_2120c_row2_col0, #T_2120c_row2_col1, #T_2120c_row2_col2, #T_2120c_row2_col15, #T_2120c_row2_col16, #T_2120c_row2_col17, #T_2120c_row3_col0, #T_2120c_row3_col15, #T_2120c_row3_col16, #T_2120c_row3_col17, #T_2120c_row4_col0, #T_2120c_row4_col6, #T_2120c_row4_col7, #T_2120c_row4_col8, #T_2120c_row4_col9, #T_2120c_row4_col10, #T_2120c_row4_col15, #T_2120c_row4_col16, #T_2120c_row4_col17, #T_2120c_row5_col0, #T_2120c_row5_col5, #T_2120c_row5_col6, #T_2120c_row5_col7, #T_2120c_row5_col8, #T_2120c_row5_col9, #T_2120c_row5_col15, #T_2120c_row5_col16, #T_2120c_row5_col17, #T_2120c_row6_col0, #T_2120c_row6_col1, #T_2120c_row6_col2, #T_2120c_row6_col3, #T_2120c_row6_col4, #T_2120c_row6_col5, #T_2120c_row6_col6, #T_2120c_row6_col7, #T_2120c_row6_col8, #T_2120c_row6_col9, #T_2120c_row6_col14, #T_2120c_row6_col15, #T_2120c_row6_col16, #T_2120c_row6_col17, #T_2120c_row7_col0, #T_2120c_row7_col1, #T_2120c_row7_col2, #T_2120c_row7_col3, #T_2120c_row7_col4, #T_2120c_row7_col5, #T_2120c_row7_col6, #T_2120c_row7_col13, #T_2120c_row7_col14, #T_2120c_row7_col15, #T_2120c_row7_col16, #T_2120c_row7_col17, #T_2120c_row8_col0, #T_2120c_row8_col1, #T_2120c_row8_col2, #T_2120c_row8_col3, #T_2120c_row8_col4, #T_2120c_row8_col13, #T_2120c_row8_col14, #T_2120c_row8_col15, #T_2120c_row8_col16, #T_2120c_row8_col17, #T_2120c_row9_col0, #T_2120c_row9_col1, #T_2120c_row9_col2, #T_2120c_row9_col3, #T_2120c_row9_col4, #T_2120c_row9_col16, #T_2120c_row9_col17, #T_2120c_row10_col0, #T_2120c_row10_col1, #T_2120c_row10_col2, #T_2120c_row10_col3, #T_2120c_row10_col4, #T_2120c_row10_col5, #T_2120c_row10_col6, #T_2120c_row10_col17 {
  font-size: 6pt;
  background-color: #ffffff;
  color: #000000;
}
#T_2120c_row1_col5 {
  font-size: 6pt;
  background-color: #efefef;
  color: #000000;
}
#T_2120c_row1_col6, #T_2120c_row1_col13 {
  font-size: 6pt;
  background-color: #7c7c7c;
  color: #f1f1f1;
}
#T_2120c_row1_col7 {
  font-size: 6pt;
  background-color: #4a4a4a;
  color: #f1f1f1;
}
#T_2120c_row1_col8, #T_2120c_row1_col9, #T_2120c_row1_col10, #T_2120c_row2_col5, #T_2120c_row2_col6, #T_2120c_row2_col7, #T_2120c_row2_col11, #T_2120c_row2_col12, #T_2120c_row2_col13, #T_2120c_row3_col4, #T_2120c_row3_col12, #T_2120c_row3_col13, #T_2120c_row4_col1, #T_2120c_row4_col2, #T_2120c_row4_col3, #T_2120c_row4_col12, #T_2120c_row4_col13, #T_2120c_row5_col12, #T_2120c_row6_col11, #T_2120c_row9_col11, #T_2120c_row10_col11, #T_2120c_row10_col12, #T_2120c_row10_col13, #T_2120c_row10_col14, #T_2120c_row10_col15, #T_2120c_row10_col16 {
  font-size: 6pt;
  background-color: #000000;
  color: #f1f1f1;
}
#T_2120c_row1_col11 {
  font-size: 6pt;
  background-color: #606060;
  color: #f1f1f1;
}
#T_2120c_row1_col12 {
  font-size: 6pt;
  background-color: #4d4d4d;
  color: #f1f1f1;
}
#T_2120c_row1_col14 {
  font-size: 6pt;
  background-color: #bbbbbb;
  color: #000000;
}
#T_2120c_row2_col3 {
  font-size: 6pt;
  background-color: #e4e4e4;
  color: #000000;
}
#T_2120c_row2_col4, #T_2120c_row8_col6 {
  font-size: 6pt;
  background-color: #6b6b6b;
  color: #f1f1f1;
}
#T_2120c_row2_col8, #T_2120c_row2_col14, #T_2120c_row3_col14 {
  font-size: 6pt;
  background-color: #171717;
  color: #f1f1f1;
}
#T_2120c_row2_col9, #T_2120c_row3_col11 {
  font-size: 6pt;
  background-color: #4b4b4b;
  color: #f1f1f1;
}
#T_2120c_row2_col10, #T_2120c_row7_col10, #T_2120c_row8_col8, #T_2120c_row8_col10, #T_2120c_row9_col8, #T_2120c_row9_col10 {
  font-size: 6pt;
  background-color: #010101;
  color: #f1f1f1;
}
#T_2120c_row3_col1 {
  font-size: 6pt;
  background-color: #272727;
  color: #f1f1f1;
}
#T_2120c_row3_col2 {
  font-size: 6pt;
  background-color: #0a0a0a;
  color: #f1f1f1;
}
#T_2120c_row3_col3 {
  font-size: 6pt;
  background-color: #050505;
  color: #f1f1f1;
}
#T_2120c_row3_col5 {
  font-size: 6pt;
  background-color: #333333;
  color: #f1f1f1;
}
#T_2120c_row3_col6 {
  font-size: 6pt;
  background-color: #e6e6e6;
  color: #000000;
}
#T_2120c_row3_col7, #T_2120c_row3_col10 {
  font-size: 6pt;
  background-color: #fafafa;
  color: #000000;
}
#T_2120c_row3_col8 {
  font-size: 6pt;
  background-color: #fbfbfb;
  color: #000000;
}
#T_2120c_row3_col9 {
  font-size: 6pt;
  background-color: #fdfdfd;
  color: #000000;
}
#T_2120c_row4_col4 {
  font-size: 6pt;
  background-color: #1b1b1b;
  color: #f1f1f1;
}
#T_2120c_row4_col5 {
  font-size: 6pt;
  background-color: #e0e0e0;
  color: #000000;
}
#T_2120c_row4_col11 {
  font-size: 6pt;
  background-color: #4e4e4e;
  color: #f1f1f1;
}
#T_2120c_row4_col14 {
  font-size: 6pt;
  background-color: #767676;
  color: #f1f1f1;
}
#T_2120c_row5_col1 {
  font-size: 6pt;
  background-color: #fcfcfc;
  color: #000000;
}
#T_2120c_row5_col2, #T_2120c_row5_col3 {
  font-size: 6pt;
  background-color: #f6f6f6;
  color: #000000;
}
#T_2120c_row5_col4, #T_2120c_row7_col7 {
  font-size: 6pt;
  background-color: #f8f8f8;
  color: #000000;
}
#T_2120c_row5_col10, #T_2120c_row10_col7 {
  font-size: 6pt;
  background-color: #e8e8e8;
  color: #000000;
}
#T_2120c_row5_col11 {
  font-size: 6pt;
  background-color: #222222;
  color: #f1f1f1;
}
#T_2120c_row5_col13, #T_2120c_row6_col12 {
  font-size: 6pt;
  background-color: #090909;
  color: #f1f1f1;
}
#T_2120c_row5_col14 {
  font-size: 6pt;
  background-color: #d0d0d0;
  color: #000000;
}
#T_2120c_row6_col10, #T_2120c_row7_col11, #T_2120c_row9_col6 {
  font-size: 6pt;
  background-color: #060606;
  color: #f1f1f1;
}
#T_2120c_row6_col13 {
  font-size: 6pt;
  background-color: #979797;
  color: #f1f1f1;
}
#T_2120c_row7_col8 {
  font-size: 6pt;
  background-color: #b6b6b6;
  color: #000000;
}
#T_2120c_row7_col9 {
  font-size: 6pt;
  background-color: #252525;
  color: #f1f1f1;
}
#T_2120c_row7_col12 {
  font-size: 6pt;
  background-color: #999999;
  color: #f1f1f1;
}
#T_2120c_row8_col5 {
  font-size: 6pt;
  background-color: #f9f9f9;
  color: #000000;
}
#T_2120c_row8_col7 {
  font-size: 6pt;
  background-color: #101010;
  color: #f1f1f1;
}
#T_2120c_row8_col9, #T_2120c_row9_col9 {
  font-size: 6pt;
  background-color: #020202;
  color: #f1f1f1;
}
#T_2120c_row8_col11 {
  font-size: 6pt;
  background-color: #545454;
  color: #f1f1f1;
}
#T_2120c_row8_col12 {
  font-size: 6pt;
  background-color: #f1f1f1;
  color: #000000;
}
#T_2120c_row9_col5 {
  font-size: 6pt;
  background-color: #f7f7f7;
  color: #000000;
}
#T_2120c_row9_col7 {
  font-size: 6pt;
  background-color: #030303;
  color: #f1f1f1;
}
#T_2120c_row9_col12 {
  font-size: 6pt;
  background-color: #181818;
  color: #f1f1f1;
}
#T_2120c_row9_col13 {
  font-size: 6pt;
  background-color: #303030;
  color: #f1f1f1;
}
#T_2120c_row9_col14 {
  font-size: 6pt;
  background-color: #a9a9a9;
  color: #f1f1f1;
}
#T_2120c_row9_col15 {
  font-size: 6pt;
  background-color: #fefefe;
  color: #000000;
}
#T_2120c_row10_col8, #T_2120c_row10_col9 {
  font-size: 6pt;
  background-color: #bababa;
  color: #000000;
}
#T_2120c_row10_col10 {
  font-size: 6pt;
  background-color: #393939;
  color: #f1f1f1;
}
</style>
<table id="T_2120c" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2120c_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_2120c_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_2120c_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_2120c_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_2120c_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_2120c_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_2120c_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_2120c_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_2120c_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_2120c_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_2120c_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_2120c_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_2120c_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_2120c_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_2120c_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_2120c_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_2120c_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_2120c_level0_col17" class="col_heading level0 col17" >17</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2120c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_2120c_row0_col0" class="data row0 col0" >0</td>
      <td id="T_2120c_row0_col1" class="data row0 col1" >0</td>
      <td id="T_2120c_row0_col2" class="data row0 col2" >0</td>
      <td id="T_2120c_row0_col3" class="data row0 col3" >0</td>
      <td id="T_2120c_row0_col4" class="data row0 col4" >0</td>
      <td id="T_2120c_row0_col5" class="data row0 col5" >0</td>
      <td id="T_2120c_row0_col6" class="data row0 col6" >0</td>
      <td id="T_2120c_row0_col7" class="data row0 col7" >0</td>
      <td id="T_2120c_row0_col8" class="data row0 col8" >0</td>
      <td id="T_2120c_row0_col9" class="data row0 col9" >0</td>
      <td id="T_2120c_row0_col10" class="data row0 col10" >0</td>
      <td id="T_2120c_row0_col11" class="data row0 col11" >0</td>
      <td id="T_2120c_row0_col12" class="data row0 col12" >0</td>
      <td id="T_2120c_row0_col13" class="data row0 col13" >0</td>
      <td id="T_2120c_row0_col14" class="data row0 col14" >0</td>
      <td id="T_2120c_row0_col15" class="data row0 col15" >0</td>
      <td id="T_2120c_row0_col16" class="data row0 col16" >0</td>
      <td id="T_2120c_row0_col17" class="data row0 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_2120c_row1_col0" class="data row1 col0" >0</td>
      <td id="T_2120c_row1_col1" class="data row1 col1" >0</td>
      <td id="T_2120c_row1_col2" class="data row1 col2" >0</td>
      <td id="T_2120c_row1_col3" class="data row1 col3" >0</td>
      <td id="T_2120c_row1_col4" class="data row1 col4" >0</td>
      <td id="T_2120c_row1_col5" class="data row1 col5" >29</td>
      <td id="T_2120c_row1_col6" class="data row1 col6" >150</td>
      <td id="T_2120c_row1_col7" class="data row1 col7" >195</td>
      <td id="T_2120c_row1_col8" class="data row1 col8" >254</td>
      <td id="T_2120c_row1_col9" class="data row1 col9" >255</td>
      <td id="T_2120c_row1_col10" class="data row1 col10" >254</td>
      <td id="T_2120c_row1_col11" class="data row1 col11" >176</td>
      <td id="T_2120c_row1_col12" class="data row1 col12" >193</td>
      <td id="T_2120c_row1_col13" class="data row1 col13" >150</td>
      <td id="T_2120c_row1_col14" class="data row1 col14" >96</td>
      <td id="T_2120c_row1_col15" class="data row1 col15" >0</td>
      <td id="T_2120c_row1_col16" class="data row1 col16" >0</td>
      <td id="T_2120c_row1_col17" class="data row1 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_2120c_row2_col0" class="data row2 col0" >0</td>
      <td id="T_2120c_row2_col1" class="data row2 col1" >0</td>
      <td id="T_2120c_row2_col2" class="data row2 col2" >0</td>
      <td id="T_2120c_row2_col3" class="data row2 col3" >48</td>
      <td id="T_2120c_row2_col4" class="data row2 col4" >166</td>
      <td id="T_2120c_row2_col5" class="data row2 col5" >224</td>
      <td id="T_2120c_row2_col6" class="data row2 col6" >253</td>
      <td id="T_2120c_row2_col7" class="data row2 col7" >253</td>
      <td id="T_2120c_row2_col8" class="data row2 col8" >234</td>
      <td id="T_2120c_row2_col9" class="data row2 col9" >196</td>
      <td id="T_2120c_row2_col10" class="data row2 col10" >253</td>
      <td id="T_2120c_row2_col11" class="data row2 col11" >253</td>
      <td id="T_2120c_row2_col12" class="data row2 col12" >253</td>
      <td id="T_2120c_row2_col13" class="data row2 col13" >253</td>
      <td id="T_2120c_row2_col14" class="data row2 col14" >233</td>
      <td id="T_2120c_row2_col15" class="data row2 col15" >0</td>
      <td id="T_2120c_row2_col16" class="data row2 col16" >0</td>
      <td id="T_2120c_row2_col17" class="data row2 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_2120c_row3_col0" class="data row3 col0" >0</td>
      <td id="T_2120c_row3_col1" class="data row3 col1" >93</td>
      <td id="T_2120c_row3_col2" class="data row3 col2" >244</td>
      <td id="T_2120c_row3_col3" class="data row3 col3" >249</td>
      <td id="T_2120c_row3_col4" class="data row3 col4" >253</td>
      <td id="T_2120c_row3_col5" class="data row3 col5" >187</td>
      <td id="T_2120c_row3_col6" class="data row3 col6" >46</td>
      <td id="T_2120c_row3_col7" class="data row3 col7" >10</td>
      <td id="T_2120c_row3_col8" class="data row3 col8" >8</td>
      <td id="T_2120c_row3_col9" class="data row3 col9" >4</td>
      <td id="T_2120c_row3_col10" class="data row3 col10" >10</td>
      <td id="T_2120c_row3_col11" class="data row3 col11" >194</td>
      <td id="T_2120c_row3_col12" class="data row3 col12" >253</td>
      <td id="T_2120c_row3_col13" class="data row3 col13" >253</td>
      <td id="T_2120c_row3_col14" class="data row3 col14" >233</td>
      <td id="T_2120c_row3_col15" class="data row3 col15" >0</td>
      <td id="T_2120c_row3_col16" class="data row3 col16" >0</td>
      <td id="T_2120c_row3_col17" class="data row3 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_2120c_row4_col0" class="data row4 col0" >0</td>
      <td id="T_2120c_row4_col1" class="data row4 col1" >107</td>
      <td id="T_2120c_row4_col2" class="data row4 col2" >253</td>
      <td id="T_2120c_row4_col3" class="data row4 col3" >253</td>
      <td id="T_2120c_row4_col4" class="data row4 col4" >230</td>
      <td id="T_2120c_row4_col5" class="data row4 col5" >48</td>
      <td id="T_2120c_row4_col6" class="data row4 col6" >0</td>
      <td id="T_2120c_row4_col7" class="data row4 col7" >0</td>
      <td id="T_2120c_row4_col8" class="data row4 col8" >0</td>
      <td id="T_2120c_row4_col9" class="data row4 col9" >0</td>
      <td id="T_2120c_row4_col10" class="data row4 col10" >0</td>
      <td id="T_2120c_row4_col11" class="data row4 col11" >192</td>
      <td id="T_2120c_row4_col12" class="data row4 col12" >253</td>
      <td id="T_2120c_row4_col13" class="data row4 col13" >253</td>
      <td id="T_2120c_row4_col14" class="data row4 col14" >156</td>
      <td id="T_2120c_row4_col15" class="data row4 col15" >0</td>
      <td id="T_2120c_row4_col16" class="data row4 col16" >0</td>
      <td id="T_2120c_row4_col17" class="data row4 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_2120c_row5_col0" class="data row5 col0" >0</td>
      <td id="T_2120c_row5_col1" class="data row5 col1" >3</td>
      <td id="T_2120c_row5_col2" class="data row5 col2" >20</td>
      <td id="T_2120c_row5_col3" class="data row5 col3" >20</td>
      <td id="T_2120c_row5_col4" class="data row5 col4" >15</td>
      <td id="T_2120c_row5_col5" class="data row5 col5" >0</td>
      <td id="T_2120c_row5_col6" class="data row5 col6" >0</td>
      <td id="T_2120c_row5_col7" class="data row5 col7" >0</td>
      <td id="T_2120c_row5_col8" class="data row5 col8" >0</td>
      <td id="T_2120c_row5_col9" class="data row5 col9" >0</td>
      <td id="T_2120c_row5_col10" class="data row5 col10" >43</td>
      <td id="T_2120c_row5_col11" class="data row5 col11" >224</td>
      <td id="T_2120c_row5_col12" class="data row5 col12" >253</td>
      <td id="T_2120c_row5_col13" class="data row5 col13" >245</td>
      <td id="T_2120c_row5_col14" class="data row5 col14" >74</td>
      <td id="T_2120c_row5_col15" class="data row5 col15" >0</td>
      <td id="T_2120c_row5_col16" class="data row5 col16" >0</td>
      <td id="T_2120c_row5_col17" class="data row5 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_2120c_row6_col0" class="data row6 col0" >0</td>
      <td id="T_2120c_row6_col1" class="data row6 col1" >0</td>
      <td id="T_2120c_row6_col2" class="data row6 col2" >0</td>
      <td id="T_2120c_row6_col3" class="data row6 col3" >0</td>
      <td id="T_2120c_row6_col4" class="data row6 col4" >0</td>
      <td id="T_2120c_row6_col5" class="data row6 col5" >0</td>
      <td id="T_2120c_row6_col6" class="data row6 col6" >0</td>
      <td id="T_2120c_row6_col7" class="data row6 col7" >0</td>
      <td id="T_2120c_row6_col8" class="data row6 col8" >0</td>
      <td id="T_2120c_row6_col9" class="data row6 col9" >0</td>
      <td id="T_2120c_row6_col10" class="data row6 col10" >249</td>
      <td id="T_2120c_row6_col11" class="data row6 col11" >253</td>
      <td id="T_2120c_row6_col12" class="data row6 col12" >245</td>
      <td id="T_2120c_row6_col13" class="data row6 col13" >126</td>
      <td id="T_2120c_row6_col14" class="data row6 col14" >0</td>
      <td id="T_2120c_row6_col15" class="data row6 col15" >0</td>
      <td id="T_2120c_row6_col16" class="data row6 col16" >0</td>
      <td id="T_2120c_row6_col17" class="data row6 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_2120c_row7_col0" class="data row7 col0" >0</td>
      <td id="T_2120c_row7_col1" class="data row7 col1" >0</td>
      <td id="T_2120c_row7_col2" class="data row7 col2" >0</td>
      <td id="T_2120c_row7_col3" class="data row7 col3" >0</td>
      <td id="T_2120c_row7_col4" class="data row7 col4" >0</td>
      <td id="T_2120c_row7_col5" class="data row7 col5" >0</td>
      <td id="T_2120c_row7_col6" class="data row7 col6" >0</td>
      <td id="T_2120c_row7_col7" class="data row7 col7" >14</td>
      <td id="T_2120c_row7_col8" class="data row7 col8" >101</td>
      <td id="T_2120c_row7_col9" class="data row7 col9" >223</td>
      <td id="T_2120c_row7_col10" class="data row7 col10" >253</td>
      <td id="T_2120c_row7_col11" class="data row7 col11" >248</td>
      <td id="T_2120c_row7_col12" class="data row7 col12" >124</td>
      <td id="T_2120c_row7_col13" class="data row7 col13" >0</td>
      <td id="T_2120c_row7_col14" class="data row7 col14" >0</td>
      <td id="T_2120c_row7_col15" class="data row7 col15" >0</td>
      <td id="T_2120c_row7_col16" class="data row7 col16" >0</td>
      <td id="T_2120c_row7_col17" class="data row7 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_2120c_row8_col0" class="data row8 col0" >0</td>
      <td id="T_2120c_row8_col1" class="data row8 col1" >0</td>
      <td id="T_2120c_row8_col2" class="data row8 col2" >0</td>
      <td id="T_2120c_row8_col3" class="data row8 col3" >0</td>
      <td id="T_2120c_row8_col4" class="data row8 col4" >0</td>
      <td id="T_2120c_row8_col5" class="data row8 col5" >11</td>
      <td id="T_2120c_row8_col6" class="data row8 col6" >166</td>
      <td id="T_2120c_row8_col7" class="data row8 col7" >239</td>
      <td id="T_2120c_row8_col8" class="data row8 col8" >253</td>
      <td id="T_2120c_row8_col9" class="data row8 col9" >253</td>
      <td id="T_2120c_row8_col10" class="data row8 col10" >253</td>
      <td id="T_2120c_row8_col11" class="data row8 col11" >187</td>
      <td id="T_2120c_row8_col12" class="data row8 col12" >30</td>
      <td id="T_2120c_row8_col13" class="data row8 col13" >0</td>
      <td id="T_2120c_row8_col14" class="data row8 col14" >0</td>
      <td id="T_2120c_row8_col15" class="data row8 col15" >0</td>
      <td id="T_2120c_row8_col16" class="data row8 col16" >0</td>
      <td id="T_2120c_row8_col17" class="data row8 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_2120c_row9_col0" class="data row9 col0" >0</td>
      <td id="T_2120c_row9_col1" class="data row9 col1" >0</td>
      <td id="T_2120c_row9_col2" class="data row9 col2" >0</td>
      <td id="T_2120c_row9_col3" class="data row9 col3" >0</td>
      <td id="T_2120c_row9_col4" class="data row9 col4" >0</td>
      <td id="T_2120c_row9_col5" class="data row9 col5" >16</td>
      <td id="T_2120c_row9_col6" class="data row9 col6" >248</td>
      <td id="T_2120c_row9_col7" class="data row9 col7" >250</td>
      <td id="T_2120c_row9_col8" class="data row9 col8" >253</td>
      <td id="T_2120c_row9_col9" class="data row9 col9" >253</td>
      <td id="T_2120c_row9_col10" class="data row9 col10" >253</td>
      <td id="T_2120c_row9_col11" class="data row9 col11" >253</td>
      <td id="T_2120c_row9_col12" class="data row9 col12" >232</td>
      <td id="T_2120c_row9_col13" class="data row9 col13" >213</td>
      <td id="T_2120c_row9_col14" class="data row9 col14" >111</td>
      <td id="T_2120c_row9_col15" class="data row9 col15" >2</td>
      <td id="T_2120c_row9_col16" class="data row9 col16" >0</td>
      <td id="T_2120c_row9_col17" class="data row9 col17" >0</td>
    </tr>
    <tr>
      <th id="T_2120c_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_2120c_row10_col0" class="data row10 col0" >0</td>
      <td id="T_2120c_row10_col1" class="data row10 col1" >0</td>
      <td id="T_2120c_row10_col2" class="data row10 col2" >0</td>
      <td id="T_2120c_row10_col3" class="data row10 col3" >0</td>
      <td id="T_2120c_row10_col4" class="data row10 col4" >0</td>
      <td id="T_2120c_row10_col5" class="data row10 col5" >0</td>
      <td id="T_2120c_row10_col6" class="data row10 col6" >0</td>
      <td id="T_2120c_row10_col7" class="data row10 col7" >43</td>
      <td id="T_2120c_row10_col8" class="data row10 col8" >98</td>
      <td id="T_2120c_row10_col9" class="data row10 col9" >98</td>
      <td id="T_2120c_row10_col10" class="data row10 col10" >208</td>
      <td id="T_2120c_row10_col11" class="data row10 col11" >253</td>
      <td id="T_2120c_row10_col12" class="data row10 col12" >253</td>
      <td id="T_2120c_row10_col13" class="data row10 col13" >253</td>
      <td id="T_2120c_row10_col14" class="data row10 col14" >253</td>
      <td id="T_2120c_row10_col15" class="data row10 col15" >187</td>
      <td id="T_2120c_row10_col16" class="data row10 col16" >22</td>
      <td id="T_2120c_row10_col17" class="data row10 col17" >0</td>
    </tr>
  </tbody>
</table>

```
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)
```
```
output :  
(6131, 6265)
```
```
show_image(three_tensors[1]);
```
output :  
![04_mnist_basics_13_0](https://github.com/tjwodud04/blog/assets/34568203/6def1657-133c-4580-bd56-a05333c68d57)  
```
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape
```
```
output :  
torch.Size([6131, 28, 28])
```
```
len(stacked_threes.shape)
```
```
output :  
3
```
```
stacked_threes.ndim
```
```
output :  
3
```
```
mean3 = stacked_threes.mean(0)
show_image(mean3);
```
![04_mnist_basics_17_0](https://github.com/tjwodud04/blog/assets/34568203/f1f2165c-f82d-44a9-b88b-33de4ad4f375)
```
mean7 = stacked_sevens.mean(0)
show_image(mean7);
```    
![04_mnist_basics_18_0](https://github.com/tjwodud04/blog/assets/34568203/c00cf046-d511-4732-9566-d59757bda335)
```
a_3 = stacked_threes[1]
show_image(a_3);
```    
![04_mnist_basics_19_0](https://github.com/tjwodud04/blog/assets/34568203/677e436b-c065-46da-b203-1facf559752e)
```
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr
```
```
output :  
(tensor(0.1114), tensor(0.2021))
```
```
dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr
```
```
output :  
(tensor(0.1586), tensor(0.3021))
```
```
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
```
```
output :  
(tensor(0.1586), tensor(0.3021))
```
```
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
```
```
arr  # numpy
```
```
output:
array([[1, 2, 3],  
       [4, 5, 6]])
```
```
tns  # pytorch
```
```
output : 
tensor([[1, 2, 3],
       [4, 5, 6]])
```
```
tns[1]
```
```
output :  
tensor([4, 5, 6])
```
```
tns[:,1]
```
```
output :  
tensor([2, 5])
```
```
tns[1,1:3]
```
```
output :  
tensor([5, 6])
```
```
tns+1
```
```
output : 
tensor([[2, 3, 4],
        [5, 6, 7]])
```
```
tns.type()
```
```
output :  
'torch.LongTensor'
```
```
tns*1.5
```
```
output:
tensor([[1.5000, 3.0000, 4.5000],
        [6.0000, 7.5000, 9.0000]]) 
```
```
valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape
```
```
output :  
(torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))
```
```
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
```
```
output :  
tensor(0.1114)
```
```
valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
```
```
output : 
(tensor([0.1161, 0.1204, 0.1193,  ..., 0.1230, 0.1424, 0.1413]), torch.Size([1010]))
```
```
tensor([1,2,3]) + tensor(1)
```
```
output :  
tensor([2, 3, 4])
```
```
(valid_3_tens-mean3).shape
```
```
output :  
torch.Size([1010, 28, 28])
```
```
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
```
```
is_3(a_3), is_3(a_3).float()
```
```
(tensor(True), tensor(1.))
```
```
is_3(valid_3_tens)
```
```
output :  
tensor([ True,  True,  True,  ...,  True, False,  True])
```
```
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
```
```
output :  
(tensor(0.9168), tensor(0.9854), tensor(0.9511))
```
```
#id gradient_descent
#caption The gradient descent process
#alt Graph showing the steps for Gradient Descent
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')
```
![04_mnist_basics_41_0](https://github.com/tjwodud04/blog/assets/34568203/352cc993-7a77-4e11-9b99-569c168c041e)
```
def f(x): return x**2
```
```
plot_function(f, 'x', 'x**2')
``` 
![04_mnist_basics_43_0](https://github.com/tjwodud04/blog/assets/34568203/a2f0c06c-c6f7-4806-b042-0dc03c2a9d6c)
```
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');
``` 
![04_mnist_basics_44_0](https://github.com/tjwodud04/blog/assets/34568203/42ea168e-8376-4a6d-8c94-a5b13d71ab14)
```
xt = tensor(3.).requires_grad_()
```
```
yt = f(xt)
yt
```
```
output :  
tensor(9., grad_fn=<PowBackward0>)
```
```
yt.backward()
```
```
xt.grad
```
```
output :  
tensor(6.)
```
```
xt = tensor([3.,4.,10.]).requires_grad_()
xt
```
```
output :  
tensor([ 3.,  4., 10.], requires_grad=True)
```
```
def f(x): return (x**2).sum()

yt = f(xt)
yt
```
```
output :  
tensor(125., grad_fn=<SumBackward0>)
```
```
yt.backward()
xt.grad
```
```
output :  
tensor([ 6.,  8., 20.])
```
```
time = torch.arange(0,20).float(); time
```
```
output :  
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
```
```
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);
```
![04_mnist_basics_53_0](https://github.com/tjwodud04/blog/assets/34568203/4a0fe394-c019-4bb8-acd0-02162e5558f7)
```
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c
```
```
def mse(preds, targets): return ((preds-targets)**2).mean()
```
```
params = torch.randn(3).requires_grad_()
```
```
#hide
orig_params = params.clone()
```
```
preds = f(time, params)
```
```
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)
```
```
show_preds(preds)
``` 
![04_mnist_basics_60_0](https://github.com/tjwodud04/blog/assets/34568203/b66dd7a1-ad1a-42b8-8051-e67819591822)
```
loss = mse(preds, speed)
loss
```
```
output :  
tensor(25823.8086, grad_fn=<MeanBackward0>)
```
```
loss.backward()
params.grad
```
```
output :  
tensor([-53195.8633,  -3419.7148,   -253.8908])
```
```
params.grad * 1e-5
```
```
output :  
tensor([-0.5320, -0.0342, -0.0025])
```
```
params
```
```
output :  
tensor([-0.7658, -0.7506,  1.3525], requires_grad=True)
```
```
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None
```
```
preds = f(time,params)
mse(preds, speed)
```
```
output :  
tensor(5435.5356, grad_fn=<MeanBackward0>)
```
```
show_preds(preds)
```
![04_mnist_basics_67_0](https://github.com/tjwodud04/blog/assets/34568203/2d0b5200-7d34-4ba0-a92d-d3298749ede7)
```
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds
```
```
for i in range(10): apply_step(params)
```
```
output :  
5435.53564453125  
1577.44921875  
847.3778076171875  
709.2225341796875  
683.0758056640625  
678.1243896484375  
677.1838989257812  
677.0023803710938  
676.9645385742188  
676.9537353515625  
```    
```
#hide
params = orig_params.detach().requires_grad_()
```
```
_,axs = plt.subplots(1,4,figsize=(12,3))
for ax in axs: show_preds(apply_step(params, False), ax)
plt.tight_layout()
``` 
![04_mnist_basics_71_0](https://github.com/tjwodud04/blog/assets/34568203/fe882884-3548-4341-af84-6c3d5713d392)
```
#hide_input
#id gradient_descent
#caption The gradient descent process
#alt Graph showing the steps for Gradient Descent
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')
```    
![04_mnist_basics_72_0](https://github.com/tjwodud04/blog/assets/34568203/f382d8f4-a6c1-41f9-874a-c76f25aaa4dc)
```
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
```
```
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
```
```
output :  
(torch.Size([12396, 784]), torch.Size([12396, 1]))
```
```
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
```
```
output :  
(torch.Size([784]), tensor([1]))
```
```
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```
```
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
```
```
weights = init_params((28*28,1))
```
```
bias = init_params(1)
```
```
(train_x[0]*weights.T).sum() + bias
```
```
output :  
tensor([20.2336], grad_fn=<AddBackward0>)`
```
```
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds
```
```
output : 
tensor([[20.2336],
        [17.0644],
        [15.2384],
        ...,
        [18.3804],
        [23.8567],
        [28.6816]], grad_fn=<AddBackward0>)
```
```
corrects = (preds>0.0).float() == train_y
corrects
```
```
output : 
tensor([[ True],
        [ True],
        [ True],
        ...,
        [False],
        [False],
        [False]])
```
```
corrects.float().mean().item()
```
```
output :  
0.4912068545818329
```
```
with torch.no_grad(): weights[0] *= 1.0001
```
```
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()
```
```
output :  
0.4912068545818329
```
```
trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])
```
```
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
```
```
torch.where(trgts==1, 1-prds, prds)
```
```
output : 
tensor([0.1000, 0.4000, 0.8000])
```
```
mnist_loss(prds,trgts)
```
```
output :  
tensor(0.4333)
```
```
mnist_loss(tensor([0.9, 0.4, 0.8]),trgts)
```
```
output :  
tensor(0.2333)
```
```
def sigmoid(x): return 1/(1+torch.exp(-x))
```
```
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
``` 
![04_mnist_basics_92_0](https://github.com/tjwodud04/blog/assets/34568203/164d604e-48d1-4158-8c26-4982a1509b3d)
```
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```
```
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
```
```
output : 
[tensor([ 3, 12,  8, 10,  2]),
 tensor([ 9,  4,  7, 14,  5]),
 tensor([ 1, 13,  0,  6, 11])]
```
```
ds = L(enumerate(string.ascii_lowercase))
ds
```
```
output :  
[(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7, 'h'),(8, 'i'),(9, 'j')...]
```
```
dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)
```
```
output : 
[(tensor([17, 18, 10, 22,  8, 14]), ('r', 's', 'k', 'w', 'i', 'o')),
 (tensor([20, 15,  9, 13, 21, 12]), ('u', 'p', 'j', 'n', 'v', 'm')),
 (tensor([ 7, 25,  6,  5, 11, 23]), ('h', 'z', 'g', 'f', 'l', 'x')),
 (tensor([ 1,  3,  0, 24, 19, 16]), ('b', 'd', 'a', 'y', 't', 'q')),
 (tensor([2, 4]), ('c', 'e'))]
```
```
weights = init_params((28*28,1))
bias = init_params(1)
```
```
dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape
```
```
output : 
(torch.Size([256, 784]), torch.Size([256, 1]))
```
```
valid_dl = DataLoader(valid_dset, batch_size=256)
```
```
batch = train_x[:4]
batch.shape
```
```
output :  
torch.Size([4, 784])
```
```
preds = linear1(batch)
preds
```
```
output : 
tensor([[-2.1876],
        [-8.3973],
        [ 2.5000],
        [-4.9473]], grad_fn=<AddBackward0>)
```
```
loss = mnist_loss(preds, train_y[:4])
loss
```
```
output :  
tensor(0.7419, grad_fn=<MeanBackward0>)
```
```
loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad
```
```
output :  
(torch.Size([784, 1]), tensor(-0.0061), tensor([-0.0420]))
```
```
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
```
```
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```
```
output :  
(tensor(-0.0121), tensor([-0.0840]))
```
```
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```
```
output :  
(tensor(-0.0182), tensor([-0.1260]))
```
```
weights.grad.zero_()
bias.grad.zero_();
```
```
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```
```
(preds>0.0).float() == train_y[:4]
```
```
output : 
tensor([[False],
        [False],
        [ True],
        [False]])
```
```
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```
```
batch_accuracy(linear1(batch), train_y[:4])
```
```
output :  
tensor(0.2500)
```
```
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```
```
validate_epoch(linear1)
```
```
output :  
0.5263
```
```
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)
```
```
output :  
0.6663
```
```
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
```
```
output :  
0.8265 0.89 0.9183 0.9276 0.9398 0.9467 0.9506 0.9525 0.956 0.9579 0.9599 0.9608 0.9613 0.9618 0.9633 0.9638 0.9647 0.9657 0.9672 0.9677
```
```
linear_model = nn.Linear(28*28,1)
```
```
w,b = linear_model.parameters()
w.shape,b.shape
```
```
output :  
(torch.Size([1, 784]), torch.Size([1]))
```
```
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```
```
opt = BasicOptim(linear_model.parameters(), lr)
```
```
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```
```
validate_epoch(linear_model)
```
```
output :  
0.4607
```
```
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
```
```
train_model(linear_model, 20)
```
```
output :  
0.4932 0.7685 0.8554 0.9135 0.9345 0.9482 0.957 0.9633 0.9658 0.9677 0.9697 0.9716 0.9736 0.9745 0.976 0.977 0.9775 0.9775 0.978 0.9785
```
```
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```
```
output :  
0.4932 0.8179 0.8496 0.914 0.9345 0.9482 0.957 0.9619 0.9658 0.9672 0.9692 0.9712 0.9741 0.9751 0.976 0.9775 0.9775 0.978 0.9785 0.979
```
```
dls = DataLoaders(dl, valid_dl)
```
```
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```
```
learn.fit(10, lr=lr)
```

<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.636709</td>
      <td>0.503144</td>
      <td>0.495584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.429828</td>
      <td>0.248517</td>
      <td>0.777233</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.161680</td>
      <td>0.155361</td>
      <td>0.861629</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.072948</td>
      <td>0.097722</td>
      <td>0.917566</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.040128</td>
      <td>0.073205</td>
      <td>0.936212</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.027210</td>
      <td>0.059466</td>
      <td>0.950442</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.021837</td>
      <td>0.050799</td>
      <td>0.957802</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.019398</td>
      <td>0.044980</td>
      <td>0.964181</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.018122</td>
      <td>0.040853</td>
      <td>0.966143</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.017330</td>
      <td>0.037788</td>
      <td>0.968106</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>

```
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```
```
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```
```
plot_function(F.relu)
```    
![04_mnist_basics_130_0](https://github.com/tjwodud04/blog/assets/34568203/bcddaf9c-a734-4c6c-8b22-aab88f66bfa8)
```
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```
```
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```
```
#hide_output
learn.fit(40, 0.1)
```

<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.333021</td>
      <td>0.396112</td>
      <td>0.512267</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152461</td>
      <td>0.235238</td>
      <td>0.797350</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.083573</td>
      <td>0.117471</td>
      <td>0.911678</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.054309</td>
      <td>0.078720</td>
      <td>0.940628</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.040829</td>
      <td>0.061228</td>
      <td>0.956330</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.034006</td>
      <td>0.051490</td>
      <td>0.963690</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.030123</td>
      <td>0.045381</td>
      <td>0.966634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.027619</td>
      <td>0.041218</td>
      <td>0.968106</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.025825</td>
      <td>0.038200</td>
      <td>0.969087</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024441</td>
      <td>0.035901</td>
      <td>0.969578</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.023321</td>
      <td>0.034082</td>
      <td>0.971541</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.022387</td>
      <td>0.032598</td>
      <td>0.972031</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.021592</td>
      <td>0.031353</td>
      <td>0.974485</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.020904</td>
      <td>0.030284</td>
      <td>0.975466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.020300</td>
      <td>0.029352</td>
      <td>0.975466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.019766</td>
      <td>0.028526</td>
      <td>0.975466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.019288</td>
      <td>0.027788</td>
      <td>0.976448</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.018857</td>
      <td>0.027124</td>
      <td>0.977429</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.018465</td>
      <td>0.026523</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.018107</td>
      <td>0.025977</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.017777</td>
      <td>0.025479</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.017473</td>
      <td>0.025022</td>
      <td>0.979392</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.017191</td>
      <td>0.024601</td>
      <td>0.980373</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.016927</td>
      <td>0.024214</td>
      <td>0.980373</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.016680</td>
      <td>0.023855</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.016449</td>
      <td>0.023521</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.016230</td>
      <td>0.023211</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.016023</td>
      <td>0.022922</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.015827</td>
      <td>0.022653</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.015641</td>
      <td>0.022401</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.015463</td>
      <td>0.022165</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.015294</td>
      <td>0.021944</td>
      <td>0.983317</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.015132</td>
      <td>0.021736</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.014977</td>
      <td>0.021541</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.014828</td>
      <td>0.021357</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.014686</td>
      <td>0.021184</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.014549</td>
      <td>0.021019</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.014417</td>
      <td>0.020864</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.014290</td>
      <td>0.020716</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.014168</td>
      <td>0.020576</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>

```
plt.plot(L(learn.recorder.values).itemgot(2));
```    
![04_mnist_basics_134_0](https://github.com/tjwodud04/blog/assets/34568203/495cb618-c4a6-4f9f-bc26-93e679c3c119)
```
learn.recorder.values[-1][2]
```
```
output :  
0.98233562707901
```
```
dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```

<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.069452</td>
      <td>0.016400</td>
      <td>0.998037</td>
      <td>00:24</td>
    </tr>
  </tbody>
</table>

### Questionnaire

1. How is a grayscale image represented on a computer? How about a color image?  
이미지는 이미지의 내용을 나타내는 픽셀 값이 있는 배열로 표현됩니다. 그레이스케일 이미지의 경우, 그레이스케일 값을 나타내는 픽셀에 256개의 정수 범위로 구성된 2차원 배열이 사용됩니다. 0 값은 흰색, 255 값은 검은색, 그 사이에는 다양한 회색조 음영을 나타냅니다. 컬러 이미지의 경우 일반적으로 3개의 컬러 채널(빨강, 녹색, 파랑)이 사용되며, 각 채널마다 별도의 256 범위 2D 배열이 사용됩니다. 픽셀 값 0은 다시 흰색을 나타내며 255는 단색 빨강, 녹색 또는 파랑을 나타냅니다. 3개의 2D 배열은 컬러 이미지를 나타내는 최종 3D 배열(랭크 3 텐서)을 형성합니다.
  

2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?  
train과 validation이라는 두 개의 하위 폴더가 있으며, 전자는 모델 훈련용 데이터를 포함하고 후자는 각 훈련 단계 후에 모델 성능을 검증하기 위한 데이터를 포함합니다. 유효성 검사 집합에서 모델을 평가하는 두 가지 목적은 다음과 같습니다.  
a) 정확도와 같이 사람이 해석할 수 있는 메트릭을 보고하고(종종 훈련에 사용되는 추상적인 손실 함수와 달리),  
b) 훈련되지 않은 데이터 집합에서 모델을 평가하여 과적합을 쉽게 감지하기 위한 것입니다(즉, 과적합 모델은 train set에서는 점점 더 잘 수행되지만 validation set에서는 점점 더 성능이 떨어집니다.).  
물론 모든 실무자가 데이터의 train/validation 분할을 직접 생성할 수도 있습니다. 공개 데이터 세트는 일반적으로 implementation/publication 간의 결과 비교를 단순화하기 위해 미리 분할됩니다.
각 하위 폴더에는 각 이미지 클래스에 대한 .jpg 파일이 포함된 두 개의 하위 폴더 3과 7이 있습니다. 이것은 사진으로 구성된 데이터 세트를 구성하는 일반적인 방법입니다. 전체 MNIST 데이터 세트의 경우 각 숫자에 대한 이미지에 대해 하나씩 총 10개의 하위 폴더가 있습니다.
  

3. Explain how the "pixel similarity" approach to classifying digits works.
"픽셀 유사성" 접근 방식에서는 식별하려는 각 클래스에 대한 원형을 생성합니다. 이 경우 3의 이미지와 7의 이미지를 구분하고자 합니다. 전형적인 3을 train set에 있는 모든 3의 픽셀 단위 평균값으로 정의합니다. 7의 경우도 마찬가지입니다. 두 가지 원형을 시각화하면 실제로는 숫자가 흐릿하게 표현된 버전임을 확인할 수 있습니다. 이전에 보이지 않던 이미지가 3인지 7인지 구분하기 위해 두 원형과의 거리를 계산합니다 (여기서는 평균 픽셀 단위의 절대적 차이). 새로운 이미지의 원형 3과의 거리가 원형 7의 두 개보다 작으면 3이라고 말합니다.
  

4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
리스트(다른 프로그래밍 언어의 배열)는 종종 for-루프를 사용하여 생성됩니다. list comprehension은 for-루프를 사용하여 목록을 생성하는 것을 단일 표현식으로 압축하는 파이썬 방식입니다. 종종 필터링을 위한 if 절도 포함됩니다.  
ex)
```python
lst_in = range(10)
lst_out = [2*el for el in lst_in if el%2==1] #list comprehension
# is equivalent to:
lst_out = []
for el in lst_in:
       if el%2==1:
       lst_out.append(2*el)
```

5. What is a "rank-3 tensor"?  
텐서의 순위는 텐서가 가진 차원 수입니다. 텐서 내에서 숫자를 참조하는 데 필요한 인덱스의 개수로 순위를 쉽게 파악할 수 있습니다. 스칼라는 0순위 텐서(인덱스 없음), 벡터는 1순위 텐서(인덱스 1개, 예: v[i]), 행렬은 2순위 텐서(인덱스 2개, 예: a[i,j]), 3순위 텐서는 직육면체 또는 "행렬의 스택"(인덱스 3개, 예: b[i,j,k])으로 나타낼 수 있습니다. 특히, 텐서의 랭크는 모양이나 차원과 무관하며, 예를 들어 2x2x2 모양의 텐서와 3x5x7 모양의 텐서는 모두 랭크 3을 갖습니다.
"랭크"라는 용어는 텐서 및 행렬(선형적으로 독립적인 열 벡터의 수를 의미)의 맥락에서 다른 의미를 가짐에 유의하세요.
  

6. What is the difference between tensor rank and shape? How do you get the rank from the shape?  
랭크는 텐서의 축 또는 차원 수이며, 모양은 텐서의 각 축의 크기입니다.
  

7. What are RMSE and L1 norm?  
L2 norm이라고도 하는 평균제곱근오차(RMSE)와 L1 norm이라고도 하는 평균절대차(MAE)는 '거리'를 측정하는 데 일반적으로 사용되는 두 가지 방법입니다. 단순한 차이는 어떤 차이는 양수이고 어떤 차이는 음수여서 서로 상쇄되기 때문에 작동하지 않습니다. 따라서 거리를 제대로 측정하려면 차이의 크기에 초점을 맞춘 함수가 필요합니다. 가장 간단한 방법은 차이의 절대값을 더하는 것인데, 이것이 바로 MAE입니다. RMSE는 제곱의 평균을 구한 다음(모든 것을 양수로 함) 제곱근을 구합니다(제곱을 되돌림).
  

8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?  
파이썬에서 루프는 매우 느리기 때문에 개별 요소를 반복하는 대신 연산을 배열 연산으로 표현하는 것이 가장 좋습니다. 이렇게 할 수 있다면 순수 Python보다 훨씬 빠른 기본 C 코드를 사용하기 때문에 NumPy 또는 PyTorch를 사용하는 것이 수천 배 더 빠릅니다. 더 좋은 점은 PyTorch를 사용하면 GPU에서 연산을 실행할 수 있으므로 병렬 연산을 수행할 수 있는 경우 속도가 크게 빨라집니다.
  

9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.  
```
    In [ ]: a = torch.Tensor(list(range(1,10))).view(3,3); print(a)
    Out [ ]: tensor([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]])
    In [ ]: b = 2 * a; print(b)
    Out [ ]: tensor([[ 2.,  4.,  6.],
                     [ 8., 10., 12.],
                     [14., 16., 18.]])
    In [ ]:  b[1:,1:]
    Out []: tensor([[10., 12.],
                    [16., 18.]])
```
  
10. What is broadcasting?  
NumPy나 PyTorch와 같은 과학/수학용 파이썬 패키지는 종종 코드를 더 쉽게 작성할 수 있는 브로드캐스팅을 구현합니다. PyTorch의 경우, 순위가 작은 텐서는 순위가 큰 텐서와 동일한 크기를 갖도록 확장됩니다. 이러한 방식으로 순위가 다른 텐서 간에 연산을 수행할 수 있습니다.
  

11. Are metrics generally calculated using the training set, or the validation set? Why?  
metric은 일반적으로 validation set에서 계산됩니다. validation set은 모델에 대해 보이지 않는 데이터이므로, 과적합이 있는지, 유사한 데이터가 주어졌을 때 모델이 얼마나 잘 일반화할 수 있는지 확인하려면 validation set에서 메트릭을 평가하는 것이 더 좋습니다.
  

12. What is SGD?  
SGD, 즉 확률적 경사 하강은 최적화 알고리즘입니다. 구체적으로 SGD는 예측 및 목표에 대해 평가된 주어진 손실 함수를 최소화하기 위해 모델의 매개변수를 업데이트하는 알고리즘입니다. SGD(그리고 많은 최적화 알고리즘)의 핵심 아이디어는 손실 함수의 기울기가 매개변수 공간에서 손실 함수가 어떻게 변화하는지를 나타내며, 이를 통해 손실 함수를 최소화하기 위해 매개변수를 업데이트하는 최선의 방법을 결정할 수 있다는 것입니다. 이것이 바로 SGD가 하는 일입니다.
  

13. Why does SGD use mini-batches?  
하나 이상의 데이터 포인트에서 손실 함수(및 그래디언트)를 계산해야 합니다. 계산 제한과 시간 제약으로 인해 전체 데이터 세트에 대해 계산할 수 없습니다. 그러나 각 데이터 포인트를 반복하면 기울기가 불안정하고 부정확해져 훈련에 적합하지 않습니다. 이를 타협하기 위해 데이터 집합의 작은 하위 집합에 대한 평균 손실을 한 번에 계산합니다. 이 하위 집합을 미니 배치라고 합니다. 또한 미니 배치를 사용하면 GPU에서 단일 항목을 사용하는 것보다 계산 효율이 더 높습니다.
  

14. What are the seven steps in SGD for machine learning?  
    (1) 매개변수 초기화 - 임의의 값이 가장 잘 작동하는 경우가 많습니다.  
    (2) 예측 계산 - 이 작업은 훈련 세트에서 한 번에 하나의 미니 배치씩 수행됩니다.  
    (3) 손실 계산 - 미니 배치에 대한 평균 손실이 계산됩니다.  
    (4) 기울기 계산 - 손실 함수를 최소화하기 위해 매개 변수를 어떻게 변경해야 하는지에 대한 근사치입니다.  
    (5) 가중치 단계 - 계산된 가중치에 따라 파라미터를 업데이트합니다.  
    (6) 이 과정을 반복합니다.  
    (7) 중지 - 실제로는 시간 제약에 따라 또는 일반적으로 훈련/검증 손실 및 지표 개선이 중단되는 시점을 기준으로 합니다.  
  

15. How do we initialize the weights in a model?    
랜덤한 가중치를 줍니다
  

16. What is "loss"?  
손실 함수는 주어진 예측과 목표에 따라 값을 반환하며, 값이 낮을수록 더 나은 모델 예측에 해당합니다.
  

17. Why can't we always use a high learning rate?  
옵티마이저가 너무 큰 조치를 취하고 매개변수를 예상보다 빠르게 업데이트하기 때문에 손실이 '튕기거나(진동)' 심지어 차이가 날 수도 있습니다.
  

18. What is a "gradient"?  
그래디언트는 모델을 개선하기 위해 각 가중치를 얼마나 변경해야 하는지 알려줍니다. 이는 본질적으로 모델의 가중치(미분)를 변경할 때 손실 함수가 어떻게 변하는지를 나타내는 척도입니다.
  

19. Do you need to know how to calculate gradients yourself?  
딥러닝 라이브러리가 자동으로 그라데이션을 계산하므로 그라데이션을 수동으로 계산할 필요가 없습니다. 이 기능을 automatic differentiation이라고 합니다. PyTorch에서 requires_grad가 True인 경우, 역방향 메서드인 a.backward()를 호출하여 그라디언트를 반환할 수 있습니다.
  

20. Why can't we use accuracy as a loss function?  
가중치를 조정할 때 손실 함수가 변경되어야 합니다. 정확도는 모델의 예측이 변경되는 경우에만 변경됩니다. 따라서 예측에 대한 신뢰도가 향상되었지만 예측이 변경되지 않는 등 모델에 약간의 변경이 있는 경우 정확도는 여전히 변경되지 않습니다. 따라서 실제 예측이 변경되는 경우를 제외하고는 모든 곳에서 기울기가 0이 됩니다. 따라서 모델은 0과 같은 기울기로부터 학습할 수 없으며, 모델의 가중치가 업데이트되지 않고 학습되지 않습니다. 좋은 손실 함수는 모델이 약간 더 나은 예측을 할 때 약간 더 나은 손실을 제공합니다. 예측이 약간 더 좋다는 것은 모델이 올바른 예측에 대해 더 확신을 가지고 있다는 것을 의미합니다. 예를 들어, MNIST 이미지가 3일 확률을 0.9 대 0.7로 예측하는 것이 약간 더 나은 예측일 수 있습니다. 손실 함수는 이를 반영해야 합니다.
  

21. Draw the sigmoid function. What is special about its shape?  
![sigmoid function](https://forums.fast.ai/uploads/default/original/3X/1/1/11fa5da9a15e9b4db282ff9bc1f8237073173e7d.png)  
시그모이드 함수는 모든 값을 0과 1 사이의 값으로 쪼개는 부드러운 곡선입니다. 대부분의 손실 함수는 모델이 0과 1 사이의 어떤 형태의 확률 또는 신뢰 수준을 출력한다고 가정하므로 이를 위해 모델의 마지막에 시그모이드 함수를 사용합니다.
  

22. What is the difference between a loss function and a metric?  
중요한 차이점은 메트릭은 인간의 이해를 유도하고 손실은 자동화된 학습을 유도한다는 점입니다. 손실이 학습에 유용하게 사용되려면 의미 있는 파생물이 있어야 합니다. 정확도와 같은 많은 메트릭은 그렇지 않습니다. 대신 메트릭은 사람이 관심을 갖는 숫자로, 모델의 성능을 반영합니다.
  

23. What is the function to calculate new weights using a learning rate?  
optimizer의 step function
  

24. What does the `DataLoader` class do?  
DataLoader 클래스는 파이썬 컬렉션을 가져와 여러 배치에 걸친 이터레이터로 변환할 수 있습니다.
  

25. Write pseudocode showing the basic steps taken in each epoch for SGD.  
```python
for x,y in dl:
     pred = model(x)
     loss = loss_func(pred, y)
     loss.backward()
     parameters -= parameters.grad * lr
```
  

26. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?  
```def func(a,b): return list(zip(a,b))```  
이 데이터 구조는 각 튜플에 입력 데이터와 레이블이 포함된 튜플 목록이 필요할 때 머신 러닝 모델에 유용합니다.
  

27. What does `view` do in PyTorch?  
텐서의 내용을 변경하지 않고 텐서의 모양을 변경합니다.
  

28. What are the "bias" parameters in a neural network? Why do we need them?    
bias 파라미터가 없으면 입력이 0일 때, 출력은 항상 0이 됩니다. 따라서 바이어스 매개변수를 사용하면 모델에 추가적인 유연성이 추가됩니다.
  

29. What does the `@` operator do in Python?    
Decorator입니다. 메인 구문이 있고, 여기에 부가적인 구문을 추가하고 싶을 때. 그리고 부가적인 구문을 반복해서 사용하고 싶은 경우도 있습니다. 이 때 부가적인(그리고 반복적인) 작업을 Decorator로 선언해서 자유롭게 사용이 가능합니다.
  

30. What does the `backward` method do?    
이 메서드는 현재 gradient를 반환합니다.
  

31. Why do we have to zero the gradients?  
PyTorch는 변수의 기울기를 이전에 저장된 기울기에 추가합니다. 기울기를 제로화하지 않고 훈련 루프 함수를 여러 번 호출하면 이전에 저장된 기울기 값에 현재 손실의 기울기가 추가됩니다.
  

32. What information do we have to pass to `Learner`?  
DataLoader, 모델, 최적화 함수, 손실 함수, 그리고 선택적으로 인쇄할 metric을 전달해야 합니다.
  
   
33. Show Python or pseudocode for the basic steps of a training loop.    
```python
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
for i in range(20):
    train_epoch(model, lr, params)
```
  

34. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.    
ReLU는 "음수를 0으로 바꾸기"를 의미합니다. 일반적으로 사용되는 활성화 함수입니다.    
![ReLU function](https://forums.fast.ai/uploads/default/original/3X/2/c/2ccd17a969b3e3547b34c5823066d5874415db71.png)
  

35. What is an "activation function"?    
활성화 함수는 신경망의 일부인 또 다른 함수로, 모델에 비선형성을 제공하는 데 목적이 있습니다. 활성화 함수가 없으면 y=mx+b 형식의 선형 함수가 여러 개 있을 뿐이라는 개념입니다. 그러나 일련의 선형 레이어는 하나의 선형 레이어와 동일하므로 모델은 데이터에 선 하나만 맞출 수 있습니다. 선형 레이어 사이에 비선형성을 도입하면 더 이상 그렇지 않습니다. 각 레이어는 나머지 레이어와 어느 정도 분리되어 있으며, 이제 모델은 훨씬 더 복잡한 함수를 맞출 수 있습니다. 실제로 이러한 모델은 올바른 가중치를 사용하여 충분히 큰 모델이라면 계산 가능한 모든 문제를 임의로 높은 정확도로 해결할 수 있다는 것을 수학적으로 증명할 수 있습니다. 이를 universal approximation theorem이라고 합니다.
  

36. What's the difference between `F.relu` and `nn.ReLU`?    
F.relu는 ReLU 활성화 함수를 위한 파이썬 함수입니다. 반면에 nn.ReLU는 PyTorch 모듈입니다.
  

37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?    
둘 이상의 비선형성을 사용하면 실질적인 성능 이점이 있습니다. 더 적은 수의 파라미터로 더 심층적인 모델을 사용할 수 있고, 성능이 향상되며, 학습 속도가 빨라지고, 컴퓨팅/메모리 요구 사항이 줄어듭니다.
  

### Further Research

1. Create your own implementation of `Learner` from scratch, based on the training loop shown in this chapter.
2. Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.
