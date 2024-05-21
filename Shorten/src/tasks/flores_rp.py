import argparse
import logging
import sys
from src.datasets.common import get_translation_from_hyp, get_spBLEU
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel
import sys
sys.path.append("/cpfs01/shared/NLP-A100/NLP-A100_hdd/jie/multilingual_LLM/scripts/trans.train/batch_train_chatglm/chatglm2-6b/")
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration

DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

NIDS={
    "5": [13, 263, 278, 297, 304, 310, 322, 338, 376, 393, 411, 29871, 29889, 29892, 29896, 29900, 29901, 29949],
    "50": [13, 263, 273, 278, 292, 297, 304, 310, 313, 322, 327, 338, 350, 363, 366, 367, 373, 376, 379, 385, 393, 408, 411, 459, 471, 505, 526, 674, 746, 785, 825, 884, 937, 963, 1122, 1244, 1253, 1274, 1312, 1372, 1423, 1497, 1568, 1576, 1919, 1938, 2230, 2238, 2533, 3144, 3282, 3645, 4505, 5153, 5642, 6388, 7466, 12544, 13018, 13310, 19573, 22994, 26686, 29595, 29871, 29879, 29889, 29890, 29891, 29892, 29896, 29899, 29900, 29901, 29908, 29915, 29937, 29945, 29949, 30010, 30098],
    "95": [1, 13, 135, 137, 152, 153, 191, 229, 262, 263, 264, 266, 267, 269, 270, 271, 273, 275, 276, 277, 278, 279, 280, 284, 285, 286, 287, 289, 290, 291, 292, 295, 297, 298, 300, 301, 302, 303, 304, 307, 310, 312, 313, 314, 315, 316, 317, 318, 319, 321, 322, 323, 324, 325, 326, 327, 330, 338, 339, 340, 341, 342, 343, 347, 348, 350, 351, 352, 358, 359, 360, 361, 362, 363, 364, 366, 367, 369, 372, 373, 374, 375, 376, 377, 378, 379, 380, 383, 384, 385, 386, 390, 392, 393, 395, 396, 398, 402, 403, 404, 406, 407, 408, 409, 410, 411, 413, 414, 415, 417, 423, 424, 425, 429, 431, 433, 435, 437, 438, 439, 440, 441, 445, 446, 447, 448, 450, 451, 454, 455, 457, 459, 465, 467, 468, 471, 476, 478, 479, 480, 481, 482, 485, 488, 489, 491, 492, 493, 495, 497, 501, 505, 506, 508, 509, 510, 512, 513, 515, 520, 521, 522, 523, 525, 526, 527, 532, 537, 540, 541, 545, 546, 547, 549, 550, 552, 554, 557, 558, 559, 560, 562, 563, 564, 566, 568, 569, 574, 577, 578, 579, 582, 583, 585, 592, 596, 598, 599, 603, 607, 610, 611, 612, 615, 616, 619, 621, 623, 624, 625, 626, 627, 634, 635, 636, 637, 643, 645, 652, 654, 658, 659, 663, 669, 670, 674, 677, 679, 680, 681, 682, 685, 687, 688, 692, 694, 697, 699, 700, 701, 704, 707, 712, 714, 715, 716, 718, 719, 723, 724, 727, 728, 735, 740, 743, 746, 748, 749, 750, 752, 756, 763, 764, 768, 769, 772, 773, 774, 776, 785, 789, 790, 791, 794, 797, 799, 800, 802, 806, 808, 812, 817, 822, 823, 825, 827, 831, 833, 837, 839, 847, 848, 853, 855, 856, 861, 864, 867, 869, 871, 873, 874, 875, 881, 882, 883, 884, 889, 892, 898, 899, 903, 905, 908, 911, 914, 917, 920, 921, 925, 926, 929, 934, 936, 937, 938, 940, 943, 945, 946, 952, 960, 963, 964, 967, 968, 970, 975, 977, 982, 990, 992, 993, 996, 997, 1001, 1006, 1007, 1008, 1016, 1017, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1029, 1030, 1032, 1033, 1036, 1040, 1041, 1048, 1049, 1054, 1056, 1058, 1063, 1066, 1075, 1076, 1079, 1080, 1082, 1089, 1090, 1100, 1101, 1102, 1105, 1111, 1113, 1121, 1122, 1123, 1135, 1137, 1139, 1141, 1144, 1147, 1152, 1153, 1156, 1160, 1161, 1162, 1166, 1168, 1171, 1172, 1173, 1175, 1179, 1181, 1183, 1195, 1196, 1201, 1207, 1208, 1209, 1213, 1218, 1220, 1222, 1228, 1233, 1234, 1235, 1236, 1238, 1242, 1244, 1245, 1250, 1253, 1259, 1266, 1269, 1270, 1274, 1278, 1281, 1283, 1284, 1286, 1294, 1295, 1299, 1306, 1308, 1309, 1310, 1311, 1312, 1317, 1321, 1326, 1333, 1334, 1336, 1343, 1348, 1350, 1352, 1358, 1363, 1364, 1367, 1371, 1372, 1375, 1379, 1384, 1391, 1394, 1395, 1398, 1401, 1407, 1409, 1416, 1423, 1424, 1426, 1431, 1432, 1434, 1437, 1454, 1455, 1457, 1460, 1461, 1466, 1468, 1469, 1472, 1474, 1475, 1480, 1489, 1494, 1497, 1505, 1506, 1510, 1513, 1522, 1523, 1525, 1531, 1535, 1539, 1540, 1546, 1547, 1550, 1553, 1555, 1558, 1560, 1565, 1568, 1575, 1576, 1583, 1584, 1590, 1592, 1594, 1597, 1600, 1602, 1603, 1605, 1623, 1624, 1625, 1634, 1639, 1642, 1644, 1646, 1650, 1653, 1657, 1664, 1674, 1682, 1690, 1691, 1693, 1696, 1699, 1708, 1727, 1728, 1734, 1741, 1745, 1746, 1749, 1750, 1754, 1757, 1763, 1774, 1775, 1776, 1783, 1784, 1788, 1790, 1794, 1795, 1796, 1797, 1809, 1813, 1815, 1816, 1820, 1821, 1823, 1827, 1831, 1833, 1834, 1838, 1845, 1846, 1847, 1848, 1854, 1855, 1858, 1860, 1862, 1867, 1869, 1871, 1873, 1875, 1878, 1881, 1883, 1886, 1896, 1897, 1898, 1906, 1912, 1915, 1919, 1923, 1925, 1932, 1933, 1936, 1938, 1950, 1953, 1955, 1960, 1971, 1973, 1974, 1975, 1981, 1987, 1994, 1996, 1997, 2000, 2002, 2008, 2010, 2022, 2025, 2030, 2034, 2038, 2041, 2047, 2059, 2084, 2085, 2086, 2093, 2099, 2106, 2111, 2113, 2117, 2125, 2130, 2133, 2134, 2138, 2139, 2147, 2153, 2154, 2158, 2166, 2173, 2174, 2175, 2177, 2181, 2182, 2183, 2187, 2189, 2198, 2201, 2208, 2209, 2212, 2215, 2216, 2217, 2218, 2221, 2224, 2228, 2230, 2235, 2238, 2242, 2243, 2244, 2247, 2253, 2258, 2261, 2264, 2266, 2274, 2276, 2277, 2287, 2288, 2289, 2290, 2291, 2292, 2296, 2300, 2301, 2308, 2319, 2326, 2329, 2330, 2331, 2335, 2337, 2343, 2348, 2354, 2355, 2360, 2362, 2364, 2366, 2367, 2369, 2371, 2373, 2379, 2381, 2395, 2400, 2401, 2415, 2418, 2423, 2428, 2429, 2439, 2450, 2453, 2463, 2465, 2470, 2472, 2478, 2479, 2481, 2482, 2486, 2496, 2499, 2501, 2506, 2507, 2511, 2532, 2533, 2540, 2541, 2543, 2545, 2548, 2567, 2576, 2578, 2580, 2587, 2596, 2602, 2606, 2609, 2610, 2620, 2625, 2626, 2627, 2629, 2637, 2643, 2645, 2649, 2664, 2667, 2669, 2672, 2678, 2684, 2691, 2692, 2702, 2706, 2710, 2714, 2722, 2723, 2735, 2737, 2748, 2749, 2750, 2752, 2755, 2758, 2760, 2761, 2766, 2770, 2774, 2784, 2791, 2794, 2795, 2799, 2803, 2806, 2813, 2820, 2821, 2823, 2828, 2831, 2834, 2841, 2849, 2855, 2859, 2864, 2867, 2872, 2888, 2891, 2894, 2909, 2910, 2911, 2912, 2919, 2924, 2925, 2929, 2930, 2933, 2947, 2957, 2963, 2966, 2967, 2969, 2975, 2978, 2982, 2983, 2984, 2990, 2996, 2997, 3001, 3005, 3006, 3008, 3013, 3026, 3028, 3030, 3031, 3035, 3036, 3043, 3047, 3050, 3058, 3067, 3079, 3087, 3092, 3095, 3096, 3097, 3099, 3106, 3107, 3111, 3112, 3113, 3115, 3118, 3120, 3121, 3131, 3132, 3136, 3137, 3138, 3144, 3145, 3147, 3151, 3158, 3159, 3163, 3168, 3173, 3183, 3184, 3192, 3196, 3200, 3203, 3206, 3209, 3210, 3211, 3226, 3228, 3236, 3237, 3241, 3244, 3249, 3250, 3266, 3268, 3269, 3274, 3276, 3282, 3290, 3297, 3300, 3301, 3305, 3307, 3308, 3322, 3323, 3335, 3338, 3349, 3352, 3367, 3368, 3380, 3391, 3395, 3402, 3407, 3412, 3414, 3415, 3421, 3423, 3427, 3430, 3440, 3454, 3458, 3466, 3470, 3474, 3480, 3481, 3488, 3492, 3495, 3496, 3500, 3508, 3509, 3521, 3525, 3528, 3529, 3538, 3544, 3546, 3554, 3563, 3573, 3575, 3578, 3587, 3592, 3596, 3600, 3617, 3618, 3623, 3624, 3625, 3629, 3631, 3633, 3644, 3645, 3647, 3650, 3654, 3656, 3661, 3664, 3667, 3673, 3687, 3689, 3690, 3696, 3697, 3699, 3704, 3710, 3714, 3717, 3719, 3720, 3730, 3737, 3745, 3750, 3755, 3762, 3763, 3773, 3775, 3785, 3787, 3791, 3794, 3796, 3800, 3801, 3818, 3820, 3824, 3826, 3837, 3838, 3839, 3841, 3847, 3850, 3856, 3860, 3864, 3869, 3880, 3887, 3893, 3900, 3904, 3907, 3921, 3926, 3933, 3939, 3946, 3950, 3952, 3964, 3965, 3970, 3979, 4004, 4013, 4019, 4023, 4033, 4038, 4040, 4048, 4052, 4056, 4066, 4067, 4070, 4071, 4078, 4080, 4082, 4087, 4098, 4099, 4105, 4116, 4123, 4124, 4125, 4129, 4135, 4148, 4150, 4158, 4168, 4169, 4183, 4187, 4192, 4195, 4196, 4203, 4216, 4231, 4233, 4240, 4241, 4242, 4249, 4251, 4253, 4272, 4274, 4294, 4297, 4298, 4300, 4306, 4307, 4323, 4335, 4344, 4346, 4347, 4360, 4362, 4366, 4367, 4368, 4377, 4382, 4402, 4404, 4405, 4410, 4414, 4424, 4425, 4430, 4433, 4434, 4439, 4445, 4446, 4459, 4495, 4497, 4500, 4505, 4515, 4520, 4525, 4538, 4546, 4548, 4550, 4557, 4565, 4579, 4583, 4586, 4587, 4591, 4600, 4605, 4612, 4622, 4628, 4629, 4631, 4646, 4653, 4655, 4670, 4671, 4684, 4687, 4688, 4690, 4692, 4696, 4697, 4700, 4702, 4720, 4721, 4723, 4726, 4729, 4740, 4743, 4759, 4760, 4771, 4780, 4788, 4792, 4800, 4806, 4815, 4818, 4822, 4823, 4824, 4828, 4832, 4838, 4845, 4857, 4870, 4876, 4893, 4908, 4910, 4912, 4916, 4922, 4924, 4931, 4940, 4945, 4948, 4955, 4956, 4963, 4966, 4972, 4997, 5004, 5005, 5012, 5025, 5028, 5037, 5054, 5059, 5060, 5062, 5066, 5073, 5075, 5076, 5078, 5086, 5120, 5141, 5142, 5143, 5147, 5148, 5153, 5155, 5162, 5164, 5166, 5169, 5171, 5184, 5186, 5191, 5192, 5193, 5198, 5214, 5215, 5220, 5223, 5227, 5232, 5260, 5272, 5279, 5287, 5292, 5300, 5305, 5306, 5309, 5315, 5318, 5327, 5328, 5329, 5331, 5332, 5348, 5349, 5367, 5372, 5381, 5382, 5386, 5391, 5402, 5403, 5409, 5417, 5424, 5427, 5430, 5432, 5438, 5441, 5445, 5461, 5462, 5464, 5475, 5480, 5501, 5505, 5520, 5521, 5526, 5533, 5544, 5549, 5550, 5556, 5569, 5574, 5578, 5594, 5606, 5607, 5614, 5617, 5618, 5622, 5641, 5642, 5643, 5648, 5650, 5652, 5655, 5659, 5661, 5662, 5675, 5680, 5682, 5685, 5690, 5704, 5720, 5721, 5722, 5727, 5735, 5742, 5747, 5764, 5769, 5776, 5777, 5778, 5779, 5786, 5791, 5794, 5796, 5807, 5810, 5821, 5828, 5834, 5835, 5839, 5846, 5853, 5863, 5866, 5868, 5870, 5882, 5888, 5890, 5921, 5929, 5934, 5941, 5954, 5957, 5967, 5972, 5974, 5988, 5995, 6001, 6003, 6015, 6017, 6021, 6023, 6040, 6041, 6053, 6055, 6059, 6060, 6068, 6092, 6098, 6107, 6114, 6116, 6130, 6132, 6133, 6140, 6146, 6150, 6152, 6154, 6166, 6169, 6171, 6172, 6182, 6193, 6199, 6202, 6207, 6210, 6218, 6219, 6232, 6237, 6246, 6251, 6263, 6270, 6275, 6282, 6285, 6295, 6296, 6298, 6315, 6323, 6329, 6339, 6350, 6359, 6363, 6376, 6381, 6388, 6394, 6398, 6400, 6407, 6411, 6415, 6421, 6423, 6424, 6439, 6453, 6456, 6458, 6464, 6480, 6483, 6489, 6492, 6493, 6494, 6496, 6501, 6504, 6505, 6514, 6520, 6523, 6534, 6548, 6559, 6567, 6568, 6572, 6577, 6578, 6584, 6585, 6586, 6592, 6599, 6613, 6614, 6617, 6621, 6631, 6632, 6667, 6668, 6670, 6674, 6679, 6683, 6686, 6689, 6709, 6710, 6726, 6728, 6732, 6743, 6747, 6751, 6762, 6770, 6787, 6804, 6809, 6815, 6824, 6830, 6832, 6836, 6837, 6843, 6852, 6879, 6894, 6901, 6909, 6931, 6946, 6948, 6949, 6953, 6958, 6970, 6975, 6993, 6994, 7005, 7013, 7019, 7026, 7055, 7056, 7101, 7113, 7115, 7120, 7133, 7136, 7143, 7147, 7171, 7173, 7178, 7180, 7182, 7192, 7195, 7197, 7225, 7232, 7241, 7243, 7254, 7258, 7271, 7277, 7311, 7314, 7316, 7341, 7352, 7358, 7359, 7370, 7371, 7381, 7392, 7410, 7419, 7428, 7429, 7436, 7444, 7458, 7466, 7470, 7475, 7480, 7481, 7484, 7504, 7535, 7539, 7540, 7548, 7567, 7568, 7575, 7580, 7582, 7586, 7600, 7602, 7606, 7609, 7622, 7627, 7631, 7634, 7636, 7653, 7656, 7669, 7687, 7689, 7695, 7698, 7714, 7719, 7736, 7740, 7748, 7761, 7773, 7783, 7786, 7794, 7795, 7799, 7800, 7803, 7807, 7845, 7849, 7853, 7856, 7862, 7863, 7881, 7897, 7907, 7934, 7945, 7953, 7967, 7995, 7997, 8000, 8002, 8004, 8010, 8020, 8024, 8040, 8042, 8047, 8051, 8078, 8079, 8084, 8098, 8112, 8119, 8120, 8126, 8139, 8152, 8155, 8171, 8182, 8221, 8225, 8241, 8247, 8251, 8256, 8265, 8277, 8281, 8309, 8316, 8326, 8329, 8333, 8338, 8357, 8359, 8362, 8370, 8372, 8378, 8379, 8388, 8389, 8413, 8439, 8442, 8476, 8491, 8492, 8496, 8497, 8509, 8519, 8528, 8543, 8566, 8578, 8580, 8594, 8618, 8619, 8625, 8628, 8630, 8633, 8643, 8644, 8645, 8652, 8654, 8666, 8690, 8697, 8711, 8714, 8717, 8718, 8734, 8740, 8743, 8748, 8753, 8772, 8789, 8790, 8817, 8822, 8828, 8829, 8840, 8852, 8879, 8886, 8889, 8890, 8895, 8898, 8900, 8908, 8913, 8924, 8927, 8931, 8938, 8939, 8944, 8952, 8977, 8979, 8998, 9025, 9048, 9051, 9078, 9080, 9097, 9098, 9103, 9126, 9129, 9132, 9134, 9138, 9157, 9176, 9177, 9202, 9204, 9214, 9230, 9232, 9238, 9243, 9260, 9267, 9269, 9279, 9307, 9311, 9312, 9360, 9365, 9368, 9379, 9382, 9403, 9404, 9410, 9413, 9425, 9444, 9449, 9460, 9468, 9469, 9483, 9485, 9486, 9500, 9501, 9526, 9531, 9541, 9548, 9564, 9580, 9584, 9613, 9619, 9633, 9654, 9659, 9669, 9670, 9687, 9703, 9725, 9736, 9747, 9755, 9758, 9763, 9764, 9777, 9788, 9813, 9820, 9827, 9828, 9833, 9837, 9841, 9850, 9861, 9878, 9881, 9913, 9921, 9922, 9928, 9929, 9939, 9943, 9947, 9953, 9963, 9969, 9980, 9984, 9985, 10012, 10014, 10039, 10081, 10083, 10090, 10095, 10112, 10122, 10123, 10124, 10146, 10173, 10188, 10204, 10208, 10225, 10231, 10240, 10257, 10259, 10266, 10293, 10301, 10310, 10318, 10327, 10335, 10340, 10364, 10372, 10381, 10384, 10388, 10395, 10401, 10403, 10423, 10425, 10446, 10458, 10462, 10470, 10495, 10501, 10502, 10504, 10534, 10548, 10564, 10565, 10567, 10578, 10597, 10611, 10623, 10625, 10638, 10653, 10655, 10657, 10672, 10679, 10680, 10682, 10685, 10694, 10696, 10708, 10712, 10714, 10750, 10752, 10754, 10772, 10776, 10781, 10783, 10794, 10799, 10811, 10835, 10844, 10866, 10878, 10894, 10902, 10904, 10920, 10933, 10934, 10940, 10943, 10947, 10952, 10963, 10978, 11005, 11008, 11013, 11015, 11049, 11051, 11054, 11062, 11064, 11086, 11098, 11099, 11111, 11119, 11126, 11131, 11147, 11172, 11175, 11188, 11199, 11201, 11211, 11233, 11254, 11261, 11272, 11278, 11284, 11299, 11308, 11326, 11340, 11360, 11363, 11380, 11382, 11388, 11390, 11407, 11460, 11465, 11480, 11483, 11499, 11504, 11520, 11531, 11536, 11549, 11552, 11557, 11569, 11580, 11592, 11621, 11624, 11637, 11644, 11654, 11660, 11662, 11665, 11706, 11708, 11760, 11785, 11794, 11829, 11840, 11872, 11885, 11896, 11921, 11928, 11950, 11952, 11961, 11969, 11975, 11978, 11979, 11990, 12015, 12021, 12024, 12027, 12028, 12050, 12066, 12070, 12074, 12080, 12105, 12107, 12112, 12132, 12151, 12195, 12203, 12214, 12220, 12239, 12283, 12302, 12312, 12337, 12341, 12354, 12374, 12377, 12390, 12392, 12413, 12415, 12416, 12417, 12449, 12461, 12475, 12500, 12521, 12529, 12530, 12534, 12535, 12544, 12547, 12559, 12562, 12565, 12584, 12598, 12610, 12616, 12618, 12621, 12642, 12652, 12654, 12672, 12689, 12709, 12711, 12717, 12722, 12734, 12753, 12785, 12789, 12808, 12818, 12821, 12833, 12838, 12840, 12844, 12853, 12862, 12864, 12871, 12873, 12890, 12906, 12923, 12932, 12935, 12946, 12949, 12959, 12966, 12981, 12990, 12992, 12995, 12996, 13005, 13006, 13015, 13018, 13039, 13047, 13052, 13055, 13059, 13072, 13104, 13112, 13119, 13144, 13160, 13163, 13171, 13173, 13182, 13221, 13240, 13260, 13282, 13283, 13284, 13291, 13302, 13309, 13310, 13356, 13387, 13389, 13395, 13404, 13407, 13415, 13435, 13443, 13444, 13457, 13460, 13468, 13480, 13485, 13496, 13520, 13555, 13568, 13577, 13587, 13601, 13611, 13631, 13667, 13672, 13699, 13723, 13748, 13750, 13771, 13779, 13790, 13797, 13814, 13818, 13834, 13840, 13851, 13857, 13866, 13873, 13894, 13900, 13933, 13953, 13962, 13973, 13974, 13982, 13985, 13997, 14000, 14012, 14021, 14029, 14038, 14049, 14062, 14063, 14076, 14084, 14112, 14123, 14139, 14150, 14154, 14157, 14169, 14170, 14171, 14175, 14183, 14187, 14195, 14206, 14214, 14274, 14282, 14299, 14333, 14338, 14356, 14377, 14378, 14380, 14396, 14401, 14415, 14416, 14433, 14436, 14445, 14451, 14467, 14481, 14494, 14496, 14509, 14511, 14517, 14582, 14588, 14606, 14623, 14650, 14653, 14668, 14671, 14679, 14682, 14703, 14724, 14744, 14746, 14750, 14783, 14830, 14831, 14849, 14856, 14862, 14870, 14873, 14877, 14919, 14944, 14959, 14993, 15007, 15028, 15030, 15033, 15040, 15041, 15044, 15050, 15056, 15058, 15064, 15073, 15087, 15089, 15107, 15121, 15155, 15202, 15205, 15251, 15278, 15301, 15319, 15331, 15332, 15334, 15341, 15362, 15373, 15383, 15385, 15392, 15416, 15458, 15476, 15488, 15495, 15509, 15519, 15523, 15531, 15538, 15551, 15597, 15603, 15610, 15640, 15678, 15709, 15723, 15729, 15753, 15774, 15805, 15836, 15850, 15862, 15870, 15878, 15899, 15937, 15954, 15964, 15982, 15996, 16017, 16044, 16060, 16082, 16089, 16096, 16100, 16103, 16113, 16116, 16120, 16129, 16138, 16152, 16172, 16188, 16200, 16221, 16235, 16246, 16250, 16277, 16286, 16297, 16326, 16359, 16366, 16367, 16370, 16376, 16377, 16393, 16410, 16413, 16429, 16430, 16497, 16498, 16526, 16528, 16543, 16561, 16600, 16612, 16652, 16656, 16658, 16669, 16679, 16688, 16691, 16706, 16721, 16731, 16741, 16760, 16768, 16778, 16786, 16791, 16812, 16842, 16855, 16890, 16892, 16894, 16914, 16923, 16938, 16947, 16958, 16964, 16995, 16999, 17011, 17018, 17052, 17055, 17065, 17073, 17096, 17099, 17101, 17113, 17122, 17152, 17161, 17167, 17186, 17202, 17204, 17210, 17223, 17233, 17240, 17242, 17249, 17268, 17319, 17321, 17334, 17343, 17354, 17366, 17394, 17410, 17415, 17422, 17439, 17451, 17454, 17463, 17470, 17487, 17488, 17503, 17530, 17536, 17545, 17557, 17564, 17570, 17587, 17626, 17649, 17654, 17655, 17661, 17680, 17687, 17698, 17728, 17729, 17768, 17770, 17793, 17797, 17803, 17818, 17819, 17823, 17832, 17840, 17842, 17852, 17854, 17874, 17900, 17946, 17951, 17999, 18012, 18014, 18016, 18032, 18055, 18057, 18061, 18066, 18085, 18089, 18103, 18104, 18152, 18232, 18236, 18243, 18263, 18274, 18290, 18320, 18322, 18333, 18346, 18363, 18372, 18392, 18403, 18423, 18452, 18462, 18463, 18474, 18488, 18492, 18502, 18504, 18523, 18550, 18565, 18568, 18577, 18598, 18613, 18619, 18650, 18691, 18703, 18741, 18750, 18760, 18771, 18783, 18788, 18792, 18815, 18818, 18835, 18837, 18873, 18879, 18902, 18930, 18937, 18953, 18982, 18983, 19013, 19040, 19055, 19066, 19128, 19173, 19179, 19183, 19184, 19214, 19245, 19320, 19336, 19367, 19383, 19431, 19486, 19493, 19520, 19540, 19544, 19573, 19615, 19618, 19632, 19641, 19651, 19698, 19716, 19730, 19732, 19754, 19756, 19785, 19804, 19824, 19840, 19850, 19854, 19858, 19860, 19864, 19875, 19888, 19934, 19950, 19963, 19967, 19994, 20016, 20024, 20026, 20057, 20082, 20115, 20170, 20186, 20187, 20191, 20233, 20299, 20301, 20330, 20342, 20377, 20389, 20397, 20399, 20408, 20447, 20471, 20475, 20510, 20511, 20512, 20521, 20531, 20532, 20554, 20591, 20616, 20629, 20654, 20662, 20674, 20676, 20695, 20699, 20700, 20714, 20727, 20761, 20790, 20834, 20841, 20863, 20873, 20876, 20887, 20924, 20957, 20976, 20979, 21000, 21003, 21005, 21027, 21033, 21039, 21049, 21054, 21061, 21071, 21100, 21109, 21115, 21148, 21159, 21160, 21188, 21194, 21217, 21283, 21312, 21321, 21322, 21344, 21407, 21411, 21443, 21485, 21507, 21551, 21571, 21580, 21603, 21611, 21625, 21648, 21649, 21709, 21714, 21732, 21749, 21767, 21778, 21797, 21831, 21861, 21863, 21886, 21897, 21929, 21930, 21936, 21947, 21974, 22017, 22020, 22044, 22052, 22056, 22086, 22092, 22094, 22145, 22165, 22181, 22182, 22188, 22229, 22235, 22243, 22268, 22286, 22309, 22345, 22367, 22387, 22399, 22403, 22420, 22422, 22435, 22473, 22600, 22602, 22603, 22623, 22659, 22684, 22720, 22739, 22752, 22764, 22802, 22810, 22825, 22832, 22917, 22941, 22959, 22960, 22994, 22995, 23038, 23051, 23057, 23087, 23116, 23119, 23120, 23123, 23133, 23144, 23153, 23181, 23196, 23197, 23215, 23261, 23286, 23289, 23295, 23323, 23329, 23332, 23338, 23344, 23355, 23377, 23383, 23388, 23407, 23451, 23459, 23483, 23490, 23517, 23532, 23554, 23562, 23600, 23609, 23626, 23633, 23642, 23643, 23673, 23686, 23721, 23732, 23752, 23753, 23764, 23786, 23794, 23824, 23829, 23841, 23851, 23852, 23878, 23880, 23888, 23900, 23920, 23926, 23935, 23940, 23958, 23980, 23986, 24020, 24023, 24040, 24062, 24067, 24079, 24092, 24099, 24103, 24105, 24124, 24130, 24137, 24182, 24185, 24187, 24191, 24211, 24231, 24261, 24298, 24328, 24354, 24386, 24413, 24418, 24422, 24432, 24476, 24518, 24528, 24550, 24560, 24567, 24589, 24622, 24656, 24719, 24725, 24732, 24748, 24771, 24775, 24779, 24795, 24801, 24826, 24879, 24899, 24948, 24993, 24995, 25005, 25008, 25013, 25036, 25039, 25049, 25096, 25161, 25187, 25211, 25224, 25234, 25241, 25252, 25260, 25271, 25273, 25275, 25296, 25335, 25404, 25409, 25443, 25462, 25464, 25466, 25527, 25535, 25540, 25554, 25559, 25590, 25616, 25633, 25639, 25657, 25692, 25702, 25719, 25720, 25752, 25778, 25782, 25826, 25831, 25900, 25945, 25967, 25999, 26051, 26062, 26068, 26077, 26080, 26086, 26122, 26148, 26163, 26177, 26191, 26222, 26230, 26232, 26318, 26319, 26354, 26370, 26371, 26377, 26379, 26429, 26441, 26473, 26526, 26565, 26584, 26603, 26604, 26606, 26626, 26636, 26655, 26671, 26686, 26694, 26784, 26801, 26819, 26824, 26839, 26856, 26858, 26899, 26911, 26926, 26942, 26987, 26989, 26996, 27008, 27012, 27039, 27041, 27072, 27091, 27111, 27144, 27162, 27201, 27217, 27250, 27269, 27284, 27288, 27316, 27317, 27320, 27333, 27356, 27357, 27358, 27361, 27363, 27386, 27389, 27398, 27407, 27409, 27446, 27455, 27478, 27479, 27482, 27508, 27552, 27576, 27597, 27606, 27609, 27628, 27688, 27725, 27729, 27744, 27746, 27754, 27760, 27777, 27782, 27784, 27788, 27796, 27797, 27858, 27901, 27943, 27946, 27966, 27988, 27994, 28015, 28042, 28099, 28133, 28146, 28173, 28179, 28189, 28235, 28238, 28240, 28256, 28283, 28308, 28313, 28420, 28450, 28454, 28470, 28503, 28523, 28539, 28543, 28558, 28569, 28575, 28592, 28601, 28641, 28644, 28645, 28657, 28663, 28672, 28681, 28684, 28688, 28728, 28737, 28844, 28848, 28862, 28893, 28943, 28961, 28977, 28996, 28999, 29023, 29027, 29046, 29117, 29130, 29131, 29156, 29203, 29214, 29224, 29263, 29319, 29360, 29383, 29413, 29418, 29441, 29442, 29549, 29585, 29595, 29648, 29664, 29675, 29691, 29748, 29811, 29857, 29871, 29872, 29874, 29875, 29876, 29877, 29878, 29879, 29882, 29884, 29885, 29887, 29888, 29889, 29890, 29891, 29892, 29894, 29896, 29898, 29899, 29900, 29901, 29902, 29903, 29906, 29907, 29908, 29909, 29911, 29915, 29916, 29918, 29920, 29923, 29929, 29930, 29931, 29933, 29936, 29937, 29938, 29939, 29941, 29943, 29945, 29946, 29947, 29949, 29950, 29953, 29956, 29963, 29967, 29968, 29973, 29979, 29980, 29983, 29987, 29991, 29993, 29994, 29999, 30001, 30010, 30015, 30030, 30064, 30073, 30098, 30175, 30179, 30181, 30191, 30211, 30361, 30474, 30736],

}

def weight_neuron_replace(run_model, base_model, layer_name, neuron_id):
    if "embed_tokens" in layer_name:
        run_model.model.embed_tokens.weight.data[neuron_id,:] = base_model.model.embed_tokens.weight.data[neuron_id,:]
        print(f"在 'embed_tokens' 层中替换神经元 {neuron_id}")
    elif "v_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].self_attn.v_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].self_attn.v_proj.weight.data[:,neuron_id]
        print(f"在 'v_proj' 层中替换神经元 {neuron_id}")
    elif "k_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].self_attn.k_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].self_attn.k_proj.weight.data[:,neuron_id]
        print(f"在 'k_proj' 层中替换神经元 {neuron_id}")
    elif "q_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].self_attn.q_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].self_attn.q_proj.weight.data[:,neuron_id]
        print(f"在 'q_proj' 层中替换神经元 {neuron_id}")
    elif "o_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].self_attn.o_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].self_attn.o_proj.weight.data[:,neuron_id]
        print(f"在 'o_proj' 层中替换神经元 {neuron_id}")
    elif "down_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].mlp.down_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].mlp.down_proj.weight.data[:,neuron_id]
        print(f"在 'down_proj' 层中替换神经元 {neuron_id}")
    elif "gate_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].mlp.gate_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].mlp.gate_proj.weight.data[:,neuron_id]
        print(f"在 'gate_proj' 层中替换神经元 {neuron_id}")
    elif "up_proj" in layer_name:
        layer_number = extract_layer_number(layer_name)
        run_model.model.layers[layer_number].mlp.up_proj.weight.data[:,neuron_id]=base_model.model.layers[layer_number].mlp.up_proj.weight.data[:,neuron_id]
        print(f"在 'up_proj' 层中替换神经元 {neuron_id}")
    else:
        print("层类型错误！", layer_name)
    return run_model


def get_embedding_tensor_and_tokenizer(model_path):
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    input_embeddings = model.get_input_embeddings()
    return input_embeddings

def update_embedding(mask, original_embedding, model):
    state_dict = model.state_dict()
    res_tensor = []
    for key, value in state_dict.items():
        if "embed_tokens" in key:
            for i, flag in enumerate(mask):
                if not flag: # 改了，加个not
                    res_tensor.append(value[i, :])
                else:
                    res_tensor.append(original_embedding.weight[i, :])
            res_tensor = torch.stack(res_tensor, dim=0)
            state_dict[key] = res_tensor
            break
    model.load_state_dict(state_dict)
    return model

def load_model(config):
    # Load checkpoints
    torch_dtype_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
    }
    model_name_or_path = config.model_path.base_model
    if 'chatglm' in model_name_or_path:
        model = ChatGLMForConditionalGeneration.from_pretrained(config.model_path.base_model, 
                                                     torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], trust_remote_code=True,
                                                     device_map="auto")
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, 
                                              torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], trust_remote_code=True,
                                             )
        print(model.hf_device_map)
        # bloom uses only fast tokenize
        to_use_fast = False
        if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
            to_use_fast = True
        tokenizer = ChatGLMTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast, trust_remote_code=True)
    else:
        # model_name_or_path = config.model_path.base_model
        model = AutoModelForCausalLM.from_pretrained(config.model_path.base_model, 
                                                     torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], #trust_remote_code=True,
                                                     device_map="auto")
        if config.model_path.lora is not None:
            model = PeftModel.from_pretrained(model, config.model_path.lora, 
                                              torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], #trust_remote_code=True,
                                             )
        print(model.hf_device_map)
        # bloom uses only fast tokenize
        to_use_fast = False
        if "bloom" in model_name_or_path or "mpt" in model_name_or_path:
            to_use_fast = True
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast) #, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and "mpt" not in model_name_or_path:
        print("添加，特殊token")
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": tokenizer.eos_token,
            }
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and "mpt" in model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "pad_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>"
            }
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # model.config.pad_token_id = tokenizer.pad_token_id
        print("mpt 添加，特殊token；模型也得加")
    if 'chatglm' in model_name_or_path:
        print(f"添加chatglm的特殊token，先查看现有的tokenizer是否支持：")
        labels="</s>"
        res = tokenizer.encode(labels, add_special_tokens=False)
        print("eos: ",res, len(res),"记录中的eos id为： ", tokenizer.eos_token_id)
        labels="<unk>"
        res = tokenizer.encode(labels, add_special_tokens=False)
        print("pad: ",res, len(res), "记录中的pad id为： ",tokenizer.pad_token_id)
        if len(res) > 1:
            print(f"现有的tokenizer不支持eos和pad，我们添加下")
            tokens = ["</s>","<unk>"]
            tokenizer.add_tokens(tokens, special_tokens=True)
            labels="</s>"
            res = tokenizer.encode(labels, add_special_tokens=False)
            print(f"添加eos和pad后，{res}")
        
        # print("")

    print("展示下，tokenizer的特殊符号id：",tokenizer.eos_token_id,tokenizer.pad_token_id, tokenizer.eos_token, tokenizer.pad_token)
    beam_size = config.generat.beam_size
    search = config.generat.search
    temperature = config.generat.temperature
    do_sample =  config.generat.do_sample
#     gen_config = GenerationConfig(temperature=temperature,
#                                   top_p=0.9,
#                                   do_sample=do_sample,
#                                   num_beams=beam_size,
#                                   max_new_tokens=256,
#                                   eos_token_id=tokenizer.eos_token_id,
#                                   pad_token=tokenizer.pad_token_id,
#                                   )

#     if search == "beam":
#         gen_config = GenerationConfig(max_new_tokens=256,
#                                       beam_size=beam_size,
#                                       eos_token_id=tokenizer.eos_token_id,
#                                       pad_token=tokenizer.pad_token_id,
#                                       )
    if search == "sample":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=0.9,
                                      do_sample=True,
                                      num_beams=beam_size,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        
        print(f"search 采用的是sample")
    elif search == "beam":
        gen_config = GenerationConfig(max_new_tokens=256,
                                      num_beams=beam_size,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        print(f"search 采用的是beam")
    else:
        raise ValueError("generat sample setting not right!")
        
    # base_model_path="/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/multilingual_LLM/models/exps_1_bilingual_10000/llama-7b-hf.en_ca_bilingual_alpaca.json.finetune"
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     torch_dtype=torch_dtype_dict[config.model_path.torch_dtype],
    # )
    # neuron_ids = NIDS[config.rpd]
    # for neuron_id in neuron_ids:
    #     model = weight_neuron_replace(model, base_model, "embed_tokens", neuron_id)
    
    
    
    # yuanshuai 代码
    base_model_path="/cpfs01/shared/NLP-A100/NLP-A100_hdd/feyuan/multilingual_LLM/models/exps_1_bilingual_10000/llama-7b-hf.en_ca_bilingual_alpaca.json.finetune"
    neuron_ids = NIDS[config.rpd]
    mask = [True if i in neuron_ids else False for i, _ in enumerate(range(len(tokenizer)))]
    original_embedding = get_embedding_tensor_and_tokenizer(base_model_path)
    model = update_embedding(mask, original_embedding, model)
        
    return model, tokenizer, gen_config



def load_generat_config(config, tokenizer):
    
    beam_size = config.generat.beam_size
    search = config.generat.search
    temperature = config.generat.temperature
    do_sample =  config.generat.do_sample
    if search == "sample":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=0.9,
                                      do_sample=True,
                                      num_beams=beam_size,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        
        print(f"search 采用的是sample")
    elif search == "beam":
        gen_config = GenerationConfig(max_new_tokens=256,
                                      num_beams=beam_size,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        print(f"search 采用的是beam")
    else:
        raise ValueError("generat sample setting not right!")
        
    return gen_config

# Post-process the output, extract translations
def post_process(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text

def chatglm_and_mpt_postprocess(text, config):
    use_model_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    if 'chatglm' in use_model_path or 'mpt' in use_model_path:
        text = text.split('</s>')[0]
        if 'chatglm' in use_model_path:
            text = text.split('</br>')[0]
        return text
    else:
        return text

def test_process(config, data_dict):
    batch_size = config.generat.batch_size
    print(f"加载模型...")
    model, tokenizer, gen_config = load_model(config)
    print(f"模型加载完成")
    
    print(f"开始生成...")
    input_data = data_dict["input_data"]
    prompt = data_dict["prompt_list"]
    input_file=data_dict["input_file"]
    print(f"准备结果记录日志...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    if config.output.subpath is not None:
        record_dir=os.path.join(record_base_path,dataset_name, config.output.subpath)
        os.makedirs(record_dir, exist_ok=True)
    
    record_lang_dir=os.path.join(record_base_path,dataset_name,config.output.subpath,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    lang_pair = config.dataset.lang_pair
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_hyp_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_.hyp")
    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo, open(output_hyp_file, 'w', encoding='utf-8') as fo2:
        for i in tqdm(range(0, len(prompt), batch_size)):
            with torch.autocast("cuda"):
                p = prompt[i:i+batch_size]
                tokenized = tokenizer(p, padding=True, return_tensors="pt")
                input_ids = tokenized.input_ids.to(model.device)
                attn_mask = tokenized.attention_mask.to(model.device)
                input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
                attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
                with torch.no_grad():
                    # , generation_config=gen_config
                    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config,
                                                   pad_token_id=tokenizer.eos_token_id)


                decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # print(generated_ids)
                # tokens = tokenizer.convert_ids_to_tokens(generated_ids[0].tolist())
                # print(tokens)
                # sys.exit(0)
                for dec, gen_ids in zip(decoded_tokens, generated_ids):
                    # dec=chatglm_and_mpt_postprocess(dec, config)
                    print(dec, file=fo, flush=True)
                    print(post_process(dec), file=fo2, flush=True)
    hyps, refs, repeat_num = get_translation_from_hyp(output_file, os.path.dirname(input_file), lang_pair)
    score = get_spBLEU(hyps, refs)
    metric_file = os.path.join(os.path.dirname(output_file), "spBLEU_summary.csv")
    if os.path.exists(metric_file):
        with open(metric_file, 'a', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    else:
        with open(metric_file, 'w', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    print("Finished!")
    
def test_process_no_model_load(config, data_dict, model, tokenizer):
    batch_size = config.generat.batch_size
    # print(f"加载模型...")
    # model, tokenizer, gen_config = load_model(config)
    # print(f"模型加载完成")
    gen_config = load_generat_config(config, tokenizer)
    
    print(f"开始生成...")
    input_data = data_dict["input_data"]
    prompt = data_dict["prompt_list"]
    input_file=data_dict["input_file"]
    print(f"准备结果记录日志...")
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    if config.output.subpath is not None:
        record_dir=os.path.join(record_base_path,dataset_name, config.output.subpath)
        os.makedirs(record_dir, exist_ok=True)
    
    record_lang_dir=os.path.join(record_base_path,dataset_name,config.output.subpath,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    lang_pair = config.dataset.lang_pair
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_hyp_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_.hyp")
    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo, open(output_hyp_file, 'w', encoding='utf-8') as fo2:
        for i in tqdm(range(0, len(prompt), batch_size)):
            p = prompt[i:i+batch_size]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.to(model.device)
            attn_mask = tokenized.attention_mask.to(model.device)
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
            with torch.no_grad():
                # , generation_config=gen_config
                generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config,
                                               pad_token_id=tokenizer.eos_token_id)


            decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # print(generated_ids)
            # tokens = tokenizer.convert_ids_to_tokens(generated_ids[0].tolist())
            # print(tokens)
            # sys.exit(0)
            for dec, gen_ids in zip(decoded_tokens, generated_ids):
                dec=chatglm_and_mpt_postprocess(dec, config)
                print(dec, file=fo, flush=True)
                print(post_process(dec), file=fo2, flush=True)
    hyps, refs, repeat_num = get_translation_from_hyp(output_file, os.path.dirname(input_file), lang_pair)
    score = get_spBLEU(hyps, refs)
    metric_file = os.path.join(os.path.dirname(output_file), "spBLEU_summary.csv")
    if os.path.exists(metric_file):
        with open(metric_file, 'a', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    else:
        with open(metric_file, 'w', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    print("Finished!")