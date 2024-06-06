# Instance Segmentation and Teeth Classification in Panoramic X-rays

This repository contains the implementation and dataset related to the paper [Instance Segmentation and Teeth Classification in Panoramic X-rays](...), submitted to Expert Systems with Applications Journal.


## Dataset

- We introduce a set of 425 panoramic X-rays with Human annotated Bounding Boxes and Polygons, the 425 images are a subset of UFBA-UESC Dental Dataset. This dataset can be extensively used for detection and segmentation tasks for Dental Panoramic X-rays. Refer to [Description](./Dataset/Dataset_description.pdf) for understanding the organisation of annotations and panoramic X-rays.


<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>32 Teeth</th>
      <th>Restoration</th>
      <th>Dental Appliance</th>
      <th>Images</th>
      <th>Used Images</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>73</td>
      <td>24</td>
    </tr>
    <tr>
      <td>2</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>220</td>
      <td>72</td>
    </tr>
    <tr>
      <td>3</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>45</td>
      <td>15</td>
    </tr>
    <tr>
      <td>4</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>140</td>
      <td>32</td>
    </tr>
    <tr>
      <td>5</td>
      <td colspan="3">Images containing dental implant</td>
      <td>120</td>
      <td>37</td>
    </tr>
    <tr>
      <td>6</td>
      <td colspan="3">Images containing more than 32 teeth</td>
      <td>170</td>
      <td>30</td>
    </tr>
    <tr>
      <td>7</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>115</td>
      <td>33</td>
    </tr>
    <tr>
      <td>8</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>457</td>
      <td>140</td>
    </tr>
    <tr>
      <td>9</td>
      <td></td>
      <td></td>
      <td>✓</td>
      <td>45</td>
      <td>7</td>
    </tr>
    <tr>
      <td>10</td>
      <td></td>
      <td></td>
      <td></td>
      <td>115</td>
      <td>35</td>
    </tr>
    <tr>
      <td><strong>Total</strong></td>
      <td></td>
      <td></td>
      <td></td>
      <td><strong>1500</strong></td>
      <td><strong>425</strong></td>
    </tr>
  </tbody>
</table>


## Teeth Numbering 
![Teeth Numbering](./imgs/det_res.png)



## Files
```bash

final.v - Contains the final implementation of all modules coresponding to the processor.

finaltb.v - Contains the test bench for testing the implementation of the processor

main.xdc - file containing the configuration to simulate the processor on a nexys4 FPGA,
```
