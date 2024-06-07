# Instance Segmentation and Teeth Classification in Panoramic X-rays

This repository contains the implementation and dataset related to the paper [Instance Segmentation and Teeth Classification in Panoramic X-rays](...), submitted to Expert Systems with Applications Journal.


## Dataset

- We introduce a set of 425 panoramic X-rays with Human annotated Bounding Boxes and Polygons, the 425 images are a subset of UFBA-UESC Dental Dataset. This dataset can be extensively used for detection and segmentation tasks for Dental Panoramic X-rays. Refer to [Description](./Dataset/Dataset_description.pdf) for understanding the organisation of annotations and panoramic X-rays. The Distribution of Categories in the dataset are metnioned in the table below.


<table style="margin-left:auto;margin-right:auto;">
  <thead>
    <tr>
      <th style="text-align:center;">Category</th>
      <th style="text-align:center;">32 Teeth</th>
      <th style="text-align:center;">Restoration</th>
      <th style="text-align:center;">Dental Appliance</th>
      <th style="text-align:center;">Images</th>
      <th style="text-align:center;">Used Images</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">1</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">73</td>
      <td style="text-align:center;">24</td>
    </tr>
    <tr>
      <td style="text-align:center;">2</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">220</td>
      <td style="text-align:center;">72</td>
    </tr>
    <tr>
      <td style="text-align:center;">3</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">45</td>
      <td style="text-align:center;">15</td>
    </tr>
    <tr>
      <td style="text-align:center;">4</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">140</td>
      <td style="text-align:center;">32</td>
    </tr>
    <tr>
      <td style="text-align:center;">5</td>
      <td colspan="3" style="text-align:center;">Images containing dental implant</td>
      <td style="text-align:center;">120</td>
      <td style="text-align:center;">37</td>
    </tr>
    <tr>
      <td style="text-align:center;">6</td>
      <td colspan="3" style="text-align:center;">Images containing more than 32 teeth</td>
      <td style="text-align:center;">170</td>
      <td style="text-align:center;">30</td>
    </tr>
    <tr>
      <td style="text-align:center;">7</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">115</td>
      <td style="text-align:center;">33</td>
    </tr>
    <tr>
      <td style="text-align:center;">8</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">457</td>
      <td style="text-align:center;">140</td>
    </tr>
    <tr>
      <td style="text-align:center;">9</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">✓</td>
      <td style="text-align:center;">45</td>
      <td style="text-align:center;">7</td>
    </tr>
    <tr>
      <td style="text-align:center;">10</td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;">115</td>
      <td style="text-align:center;">35</td>
    </tr>
    <tr>
      <td style="text-align:center;"><strong>Total</strong></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"></td>
      <td style="text-align:center;"><strong>1500</strong></td>
      <td style="text-align:center;"><strong>425</strong></td>
    </tr>
  </tbody>
</table>


## Results

- Teeth Numbering Results 

<table>
  <tr>
    <th>Model Architecture</th>
    <th>mAP</th>
    <th>AP50</th>
  </tr>
  <tr>
    <td>Mask R-CNN[20]</td>
    <td>70.5</td>
    <td>97.2</td>
  </tr>
  <tr>
    <td>Mask R-CNN + FCN[21]</td>
    <td>74.1</td>
    <td>92.8</td>
  </tr>
  <tr>
    <td>Mask R-CNN + pointRend[21]</td>
    <td>75.3</td>
    <td>94.4</td>
  </tr>
  <tr>
    <td>PANet[20]</td>
    <td>74.0</td>
    <td>99.7</td>
  </tr>
  <tr>
    <td>HTC[20]</td>
    <td>71.1</td>
    <td>97.3</td>
  </tr>
  <tr>
    <td>ResNeSt[20]</td>
    <td>72.1</td>
    <td>96.8</td>
  </tr>
  <tr>
    <td>YOLOv8</td>
    <td>72.9</td>
    <td>94.6</td>
  </tr>
</table>

- Instance Segmentation Results

<table>
  <tr>
    <th>Model Architecture</th>
    <th>Incisors</th>
    <th>Canines</th>
    <th>Premolars</th>
    <th>Molars</th>
  </tr>
  <tr>
    <td>U-Net</td>
    <td>73.29</td>
    <td>69.92</td>
    <td>67.62</td>
    <td>64.98</td>
  </tr>
  <tr>
    <td>Mask R-CNN [33]</td>
    <td>89.56</td>
    <td>89.45</td>
    <td>88.70</td>
    <td>87.55</td>
  </tr>
  <tr>
    <td>U-Net + Mask R-CNN [33]</td>
    <td>91.55</td>
    <td>91.00</td>
    <td>90.00</td>
    <td>88.58</td>
  </tr>
  <tr>
    <td>BB-UNet + YOLOv8 ( Test Dataset 1)</td>
    <td>85.81</td>
    <td>84.91</td>
    <td>84.89</td>
    <td>84.40</td>
  </tr>
  <tr>
    <td>BB-UNet + YOLOv8 ( Test Dataset 2)</td>
    <td>85.71</td>
    <td>86.64</td>
    <td>86.22</td>
    <td>86.03</td>
  </tr>
</table>

## Teeth Numbering 
![Teeth Numbering](./imgs/det_res.png)

## Segmentation Masks
![Segmentation Masks](./imgs/seg_res.png)



## Files
```bash

final.v - Contains the final implementation of all modules coresponding to the processor.

finaltb.v - Contains the test bench for testing the implementation of the processor

main.xdc - file containing the configuration to simulate the processor on a nexys4 FPGA,
```
