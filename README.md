# Periodic Attention-based Stacked Sequence to Sequence Framework
This repository is the official implementation of [Periodic Attention-based Stacked Sequence to Sequence framework for long-term travel time prediction](https://www.sciencedirect.com/science/article/pii/S0950705122010693?utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_medium=email&utm_acid=265875295&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&CMX_ID=&utm_in=DM307246&utm_source=AC_), published in Knowledge-Based Systems. Feel free to contact me via this email (damon882046.c@nycu.edu.tw) if you have any problems.

### Guideline:
get .npy data
Implement Python on Data/files sequentially:
1. download_traffic_data.py
2. traffic_xml_to_csv.py
3. generate_traffic_data_all_road.py
4. csvraw2dataframe.py
5. csv2npy.py
   
And then implement the training process...

### If you find this code helpful, feel free to cite our paper:
```
@article{dai2022periodic,
  title={Periodic Attention-based Stacked Sequence to Sequence framework for long-term travel time prediction},
  author={Huang, Yu and Dai, Hao and Tseng, Vincent S},
  journal={Knowledge-Based Systems},
  volume={258},
  pages={109976},
  year={2022},
  publisher={Elsevier}
}
```
