import os
from typing import List

from .blended_mvs_utils import (
    build_list,
    datapath_files,
    read_cam_file,
    read_depth,
    read_depth_mask,
)

__all__ = [
    "build_list",
    "datapath_files",
    "read_cam_file",
    "read_depth",
    "read_depth_mask",
    "train_scans",
    "val_scans",
    "test_scans",
]


def train_scans() -> List[str]:
    return [
        "000000000000000000000000",
        "000000000000000000000001",
        "000000000000000000000002",
        "000000000000000000000003",
        "000000000000000000000004",
        "000000000000000000000005",
        "000000000000000000000006",
        "000000000000000000000007",
        "000000000000000000000008",
        "000000000000000000000009",
        "00000000000000000000000a",
        "00000000000000000000000b",
        "00000000000000000000000c",
        "00000000000000000000000d",
        "00000000000000000000000e",
        "00000000000000000000000f",
        "000000000000000000000010",
        "000000000000000000000011",
        "000000000000000000000012",
        "000000000000000000000015",
        "000000000000000000000016",
        "000000000000000000000017",
        "000000000000000000000018",
        "000000000000000000000019",
        "00000000000000000000001a",
        "00000000000000000000001b",
        "00000000000000000000001d",
        "5643df56138263b51db1b5f3",
        "5644bdac138263b51db9f669",
        "564a27b26d07883f460d8ab0",
        "565fb1dead14d4154dae2b94",
        "567884f58d2828b95e3c8eba",
        "567a0fb0a825d2fb79ac9a20",
        "5692a4c2adafac1f14201821",
        "569b92eb826bcba945ca002b",
        "56d73ba74bd29b8c35abade2",
        "56f34064e296120e10484dc4",
        "57102be2877e1421026358af",
        "57153d4031bb9900425bde85",
        "57177cd7fb8d93461afc4527",
        "576fefa017ce5a16397e87fd",
        "57a4a7bb6b9272286e26dc18",
        "57f8d9bbe73f6760f10e916a",
        "5841206219d291325678ca90",
        "58497cdf97b73e0b090c4273",
        "584a7333fe3cb463906c9fe6",
        "584aa8e9fe3cb463906cc7d0",
        "584ad76bfe3cb463906ce6dc",
        "584af003fe3cb463906d0e9b",
        "584b671f7072670e72bfaaf8",
        "584b81747072670e72bfbbfd",
        "584b9a747072670e72bfc49d",
        "584ba35f7072670e72bfca4d",
        "584ba5977072670e72bfcc2d",
        "584bc3997072670e72bfe58d",
        "584bc4407072670e72bfe665",
        "584bc53c7072670e72bfe85f",
        "584bd5587072670e72bffe39",
        "584bdadf7072670e72c0005c",
        "584be5ed7072670e72c007b3",
        "584c58b77072670e72c03990",
        "584c9ad27072670e72c060c5",
        "584c9cc67072670e72c063a1",
        "584cea557072670e72c07fb4",
        "584d19d47072670e72c0c6c0",
        "584dfe467072670e72c1665a",
        "584e05667072670e72c17167",
        "584e875c7072670e72c1ec94",
        "584f94e87072670e72c2d3f7",
        "584fdffd7072670e72c32dc7",
        "584fe07f7072670e72c32e59",
        "58500b007072670e72c35588",
        "5850d4f97072670e72c425d6",
        "58510bf97072670e72c46ddf",
        "5851165f7072670e72c4860d",
        "585203546789802282f2aaf5",
        "58522bd56789802282f2ecb3",
        "58524a080e7012308944bcbf",
        "58524a2e0e7012308944bcf3",
        "58524c1d0e7012308944bfda",
        "58524f170e7012308944c200",
        "585289980e7012308945276a",
        "58529a4e0e70123089454c6f",
        "585369770e7012308945c709",
        "585373640e7012308945cab9",
        "5854c405804be105852330fe",
        "58551bdf804be1058523556d",
        "585559d9804be10585238ddf",
        "5855a4fc804be1058523bd75",
        "58563650804be1058523da55",
        "58564084804be1058523e116",
        "58568c9a804be10585240b03",
        "5856ac15804be105852419d8",
        "5856ae8b804be10585241bae",
        "5856b460804be10585242059",
        "58574b35804be105852455fd",
        "58577c60b338a62ad5ff1564",
        "5857aa5ab338a62ad5ff4dbe",
        "5857acf8b338a62ad5ff5107",
        "585834cdb338a62ad5ffab4d",
        "58586810b338a62ad5ffc20c",
        "5858db6cb338a62ad500103b",
        "5858dbcab338a62ad5001081",
        "58592046b338a62ad5006b33",
        "58592854b338a62ad500750a",
        "58592d69b338a62ad5007a74",
        "5859341ab338a62ad500848d",
        "58596531b338a62ad500aace",
        "58598db2b338a62ad500bc38",
        "5859d84fb338a62ad500e5cf",
        "585a206ab338a62ad501298f",
        "585a217cb338a62ad5012b38",
        "585a2a71b338a62ad50138dc",
        "585b34afb338a62ad501e836",
        "585bb25fc49c8507c3ce7812",
        "585bbe55c49c8507c3ce81cd",
        "585d6c8a2a57cc11d4920a1e",
        "585e34302a57cc11d492be30",
        "585e54c72a57cc11d492f71a",
        "585ee0632a57cc11d4933608",
        "585f9661712e2761468dabca",
        "585ffe9a712e2761468df643",
        "586082d8712e2761468e2877",
        "586133c2712e2761468ecfe3",
        "5861d8ea712e2761468f3cb3",
        "5862388b712e2761468f84aa",
        "58625f42712e2761468fb44c",
        "586281d2712e2761468fcaa2",
        "586316e5712e276146903c4d",
        "586326ad712e276146904571",
        "58636467712e27614690661f",
        "586375c9712e276146907429",
        "586389c9712e276146908da6",
        "5863915b712e276146909135",
        "5863edf8712e27614690cce0",
        "58647495712e27614690f36d",
        "586496fa712e2761469108e7",
        "5864a935712e2761469111b4",
        "5864b076712e27614691197e",
        "5864da88712e276146913d8b",
        "58651bcc712e2761469166dc",
        "58654563712e276146918643",
        "5865f4a8712e27614691e39b",
        "58660e79712e27614691fe3d",
        "58664251712e276146923738",
        "5866445b712e27614692383e",
        "5866500d712e2761469240fd",
        "586669c6712e27614692597a",
        "58669aad712e27614692834c",
        "58669c02712e27614692851a",
        "58676c36833dfe3f7b88b7f2",
        "5867785a833dfe3f7b88c764",
        "58678b2d833dfe3f7b88e244",
        "5867969c833dfe3f7b88e8bc",
        "5867a434833dfe3f7b88edaf",
        "5868040c833dfe3f7b8934f7",
        "5868cd15833dfe3f7b89bfa3",
        "586913a49d1b5e34c2808b02",
        "586922da9d1b5e34c2809ff3",
        "586a37ec9d1b5e34c28184fc",
        "586a515a9d1b5e34c281b431",
        "586a94939d1b5e34c2823b5d",
        "586abc689d1b5e34c2826360",
        "586b0e219d1b5e34c2828862",
        "586b3db89d1b5e34c282cd52",
        "586b4c459d1b5e34c282e66d",
        "586b7d7d9d1b5e34c283359e",
        "586b8f149d1b5e34c283497c",
        "586b8f629d1b5e34c28349d6",
        "586c48329d1b5e34c2838e80",
        "586c4c4d9d1b5e34c28391a1",
        "586c5b5b9d1b5e34c2839a5b",
        "586c9fdf9d1b5e34c283b657",
        "586caab99d1b5e34c283c213",
        "586cd0779d1b5e34c28403a7",
        "586d07869d1b5e34c2842e5b",
        "586d27489d1b5e34c28453af",
        "586d55af9d1b5e34c284a999",
        "586d6d249d1b5e34c284b80e",
        "586d8a029d1b5e34c284c948",
        "586df9849d1b5e34c28506de",
        "586e279c9d1b5e34c2852180",
        "58790c82ce911104a3467c88",
        "587bc5ec2366dd5d06e262c1",
        "587c03f12366dd5d06e27722",
        "587c19da2366dd5d06e2877b",
        "587c1abf2366dd5d06e28901",
        "587c31b92366dd5d06e2a9dc",
        "587c45192366dd5d06e2c0eb",
        "587c87d02366dd5d06e2f989",
        "587c97a52366dd5d06e30a96",
        "587cec702366dd5d06e37862",
        "587cef0a2366dd5d06e379e3",
        "587db5872366dd5d06e3e0af",
        "587e2b1d2366dd5d06e41af0",
        "587e2ea62366dd5d06e41f2e",
        "587e5cb52366dd5d06e4486e",
        "587eb1822366dd5d06e45f29",
        "587f365d2366dd5d06e4906e",
        "58800b0b2366dd5d06e5312d",
        "58805eac2366dd5d06e56460",
        "5880675a2366dd5d06e570ca",
        "58806e422366dd5d06e57bb6",
        "588084032366dd5d06e59e82",
        "5880b3692366dd5d06e5d534",
        "5880e3422366dd5d06e5ff8e",
        "5880f0ef2366dd5d06e6166e",
        "588159582366dd5d06e66877",
        "588185d8dfb7a15588a114a3",
        "58818685dfb7a15588a11626",
        "5881d2bfb6844814c136a119",
        "5881f11d8ce2c2754d0714c3",
        "5881fee18ce2c2754d0723f8",
        "588230658ce2c2754d076728",
        "5882372c8ce2c2754d076af0",
        "58829563f42b1d3ee3ec835f",
        "5882cda2b116682b4adebd25",
        "5882d58fb116682b4adec7db",
        "588305ed0db9bf59bf8a8c80",
        "588315c60db9bf59bf8aa928",
        "58831d060db9bf59bf8ab98b",
        "588332ee0db9bf59bf8ae9c3",
        "5883535e932ba84fbed5ad07",
        "588457b8932ba84fbed69942",
        "5884c256932ba84fbed70bf5",
        "5884cc13932ba84fbed71ec4",
        "588519d5932ba84fbed7a04a",
        "58851ebb932ba84fbed7abad",
        "5885bc5296fa095e0671a7f0",
        "5886d14cb791366d617a362c",
        "58871dc3b791366d617a55ff",
        "58873cabb791366d617a65a7",
        "58873d44b791366d617a65dd",
        "588824d1b791366d617adeef",
        "5888358cb791366d617af69d",
        "588857f6c02346100f4ac09f",
        "58888b3dc02346100f4af665",
        "5888becfc02346100f4b0b21",
        "5888e408c02346100f4b1a29",
        "58894345c02346100f4b51ca",
        "58897f62c02346100f4b8ee6",
        "5889da66ec4d5a1c088e5187",
        "5889e344ec4d5a1c088e59be",
        "5889e754ec4d5a1c088e60ba",
        "588a34cfec4d5a1c088ea8d1",
        "588a9c5fec4d5a1c088ec350",
        "588ab5bdec4d5a1c088ed60f",
        "588aff9d90414422fbe7885a",
        "588b20d290414422fbe79f40",
        "588c08d590414422fbe8200b",
        "588c203d90414422fbe8319e",
        "588c989a90414422fbe86d96",
        "588ca09d90414422fbe871a1",
        "588cce2190414422fbe88520",
        "588cd5ef90414422fbe8875c",
        "588cf0ad90414422fbe8a20f",
        "588e01c490414422fbe8ee2a",
        "588e0d8c90414422fbe8f8b2",
        "588e35e690414422fbe90a53",
        "588f017e90414422fbe9b74b",
        "588f095190414422fbe9c1ee",
        "5890279190414422fbea9734",
        "5890330d90414422fbeaa0cb",
        "5890523090414422fbeab3f0",
        "5890641690414422fbeabbe7",
        "5890c16b90414422fbeb0262",
        "589145ef90414422fbeb2e08",
        "5891d0479a8c0314c5cd2abd",
        "5891d8ae9a8c0314c5cd30ab",
        "5891ecf19a8c0314c5cd490a",
        "5892c0cd9a8c0314c5cdc977",
        "58933bac9a8c0314c5ce3508",
        "589388059a8c0314c5ce718b",
        "58938e6d9a8c0314c5ce726f",
        "589433fa9a8c0314c5ce9656",
        "5894ab309a8c0314c5cee57d",
        "58951cb49a8c0314c5cf4d5e",
        "5895a6a89a8c0314c5cfca7c",
        "5895b8c29a8c0314c5cfd051",
        "5895d38f9a8c0314c5cfe50c",
        "5895f2329a8c0314c5d00117",
        "5896bb989a8c0314c5d086b6",
        "5896ebf39a8c0314c5d0a8c4",
        "5897076e9a8c0314c5d0d31b",
        "58970fd09a8c0314c5d0e383",
        "589765d39a8c0314c5d16b12",
        "58977ef09a8c0314c5d17b26",
        "5898b1bac9dccc22987b7f74",
        "5898b31cc9dccc22987b82ec",
        "5898b6ffc9dccc22987b8a03",
        "5898bbaac9dccc22987b8eba",
        "5899cfa6b76d7a3780a4cb64",
        "5899e5dcb76d7a3780a4ecc1",
        "589aca717dc3d323d55671c4",
        "589af2c97dc3d323d55691e8",
        "589b04287dc3d323d556a185",
        "589b49ea7dc3d323d556d9b4",
        "589bf6a57dc3d323d55743ab",
        "589c24527dc3d323d5577126",
        "589c300f7dc3d323d5577926",
        "589c35457dc3d323d5577d8d",
        "589c3c497dc3d323d5578468",
        "589c3c577dc3d323d5578480",
        "589ca6a6b896147a1b73aff7",
        "589d1e1fb896147a1b73ee5b",
        "589d5c58b896147a1b742256",
        "589d95538fa2cf375df3317b",
        "589df0ffb504a864ad63521a",
        "589ea316b504a864ad639a2b",
        "589ec97cb504a864ad63adc3",
        "589f214338486e3c9846f123",
        "589fdfe738486e3c984736cf",
        "58a01dea38486e3c98475871",
        "58a0365e38486e3c984783eb",
        "58a07ce53d0b45424799fdde",
        "58a07f233d0b45424799ffe7",
        "58a0a2f33d0b4542479a11b1",
        "58a0dd1a3d0b4542479a28f3",
        "58a160983d0b4542479a7347",
        "58a164f73d0b4542479a7a8e",
        "58a186444a4d262a170ae3ae",
        "58a1a7914a4d262a170b1101",
        "58a1bc804a4d262a170b2f01",
        "58a1d9d14a4d262a170b58fe",
        "58a1f5d74a4d262a170b65fc",
        "58a285424a4d262a170baf3e",
        "58a2a09e156b87103d3d668c",
        "58a2d9c3156b87103d3da90f",
        "58a3ccb0156b87103d3e4332",
        "58a3f2f8156b87103d3e5838",
        "58a3f6c0156b87103d3e5971",
        "58a3fc95156b87103d3e5d9b",
        "58a41819156b87103d3e92a5",
        "58a439cf156b87103d3ec885",
        "58a44463156b87103d3ed45e",
        "58a4452f156b87103d3ed55b",
        "58a44df2156b87103d3ee239",
        "58a464aa156b87103d3eec04",
        "58a47552156b87103d3f00a4",
        "58c4bb4f4a69c55606122be4",
        "58c6451e4a69c556061894f1",
        "58ca7014affdfd07c70a95ce",
        "58cf4771d0f5fb221defe6da",
        "58d36897f387231e6c929903",
        "58eaf1513353456af3a1682a",
        "58f73e7c9f5b56478738929f",
        "58f7f7299f5b5647873cb110",
        "59056e6760bb961de55f3501",
        "59071f2e5a6dbd3af4130f98",
        "590c2d70336bb52a190be886",
        "590f91851225725be9e25d4e",
        "59102c811225725be9e64149",
        "591a467a6109e14d4f09b776",
        "591cf3033162411cf9047f37",
        "591ea44850991c70dc99a207",
        "59338e76772c3e6384afbb15",
        "59350ca084b7f26bf5ce6eb8",
        "59397e493a87372f2c9e882b",
        "5940564ec2d9527ab869f7e2",
        "5947719bf1b45630bd096665",
        "5947b62af1b45630bd0c2a02",
        "5948194ff1b45630bd0f47e3",
        "5950206a41b158666ac50506",
        "59521e0b9096412211c2aa9d",
        "595979485ec6a95e86a58c8d",
        "5983012d1bd4b175e70c985a",
        "599aa591d5b41f366fed0d58",
        "59a452bf9b460239aa5d1c72",
        "59a8f851597729752c31e7e0",
        "59a9619a825418241fb88191",
        "59acd2f4b891807f439c8992",
        "59bf97fe7e7b31545da34439",
        "59c1c3e2fd6e3d4ead9f1013",
        "59da1fb88a126011d0394ae9",
        "59e75a2ca9e91f2c5526005d",
        "59e864b2a9e91f2c5529325f",
        "59ecfd02e225f6492d20fcc9",
        "59f363a8b45be22330016cad",
        "59f37f74b45be2233001ba18",
        "59f70ab1e5c5d366af29bf3e",
        "59f87d0bfa6280566fb38c9a",
        "5a0271884e62597cdee0d0eb",
        "5a03e732454a8a7ec672776c",
        "5a2a95f032a1c655cfe3de62",
        "5a2af22b32a1c655cfe46013",
        "5a2ba6de32a1c655cfe51b79",
        "5a355c271b63f53d5970f362",
        "5a3b9731e24cd76dad1a5f1b",
        "5a3ca9cb270f0e3f14d0eddb",
        "5a3cb4e4270f0e3f14d12f43",
        "5a3f4aba5889373fbbc5d3b5",
        "5a489fb1c7dab83a7d7b1070",
        # "5a48ba95c7dab83a7d7b44ed",
        "5a48c4e9c7dab83a7d7b5cc7",
        "5a48d4b2c7dab83a7d7b9851",
        "5a4a38dad38c8a075495b5d2",
        "5a533e8034d7582116e34209",
        "5a562fc7425d0f5186314725",
        "5a563183425d0f5186314855",
        "5a572fd9fc597b0478a81d14",
        "5a57542f333d180827dfc132",
        "5a588a8193ac3d233f77fbca",
        "5a5a1e48d62c7a12d5d00e47",
        "5a618c72784780334bc1972d",
        "5a6464143d809f1d8208c43c",
        "5a69c47d0d5d0a7f3b2e9752",
        "5a6b1c418d100c2f8fdc4411",
        "5a6feeb54a7fbc3f874f9db7",
        "5a752d42acc41e2423f17674",
        "5a77b46b318efe6c6736e68a",
        "5a7cb1d6fe5c0d6fb53e64fb",
        # "5a7d3db14989e929563eb153",
        "5a8315f624b8e938486e0bd8",
        "5a8aa0fab18050187cbe060e",
        "5a969eea91dfc339a9a3ad2c",
        "5a9e5df65baeef72b4a021cd",
        "5aa0f478a9efce63548c1cb4",
        # "5aa0f9d7a9efce63548c69a1",
        "5aa1196ea9efce63548ed649",
        "5aa235f64a17b335eeaf9609",
        "5aa515e613d42d091d29d300",
        "5aa7db90bfdd572271e95246",
        "5aaadd4cbc13235570d178a7",
        "5ab6af12ac4291329b1072ab",
        "5ab74bf2ac4291329b11e879",
        "5ab7e00aac4291329b15864d",
        "5ab85f1dac4291329b17cb50",
        "5ab8713ba3799a1d138bd69a",
        "5ab8b8e029f5351f7f2ccf59",
        "5abc2506b53b042ead637d86",
        "5acc7459a7853c4b5ebbef59",
        "5acf8ca0f3d8a750097e4b15",
        "5adc6bd52430a05ecb2ffb85",
        "5ae2e9c5fe405c5076abc6b2",
        # "5af02e904c8216544b4ab5a2",
        "5af28cea59bc705737003253",
        "5af545d0559359053d25dcf5",
        "5afacb69ab00705d0cefdd5b",
        "5b08286b2775267d5b0634ba",
        "5b192eb2170cf166458ff886",
        # "5b21e18c58e2823a67a10dd8",
        "5b22269758e2823a67a3bd03",
        "5b271079e0878c3816dacca4",
        "5b2c67b5e0878c381608b8d8",
        "5b37189a35304b6f75e7583e",
        "5b3b2b9e8d46a939f933fdc0",
        "5b3b353d8d46a939f93524b9",
        "5b4933abf2b5f44e95de482a",
        "5b558a928bbfb62204e77ba2",
        "5b60fa0c764f146feef84df0",
        "5b62647143840965efc0dbde",
        "5b69cc0cb44b61786eb959bf",
        "5b6e716d67b396324c2d77cb",
        "5b6eff8b67b396324c5b2672",
        "5b78e57afc8fcf6781d0c3ba",
        "5b864d850d072a699b32f4ae",
        "5b908d3dc6ab78485f3d24a9",
        "5ba75d79d76ffa2c86cf2f05",
        # "5bb7a08aea1cfa39f1a947ab",
        "5bb8a49aea1cfa39f1aa7f75",
        "5bbb6eb2ea1cfa39f1af7e0c",
        "5bc5f0e896b66a2cd8f9bd36",
        "5bccd6beca24970bce448134",
        "5bce7ac9ca24970bce4934b6",
        "5bcf979a6d5f586b95c258cd",
        "5bd43b4ba6b28b1ee86b92dd",
        "5be3a5fb8cfdd56947f6b67c",
        "5be3ae47f44e235bdbbc9771",
        "5be47bf9b18881428d8fbc1d",
        # "5be4ab93870d330ff2dce134",
        # "5be883a4f98cee15019d5b83",
        "5bea87f4abd34c35e1860ab5",
        "5beb6e66abd34c35e18e66b9",
        "5bf03590d4392319481971dc",
        "5bf17c0fd439231948355385",
        "5bf18642c50e6f7f8bdbd492",
        "5bf21799d43923194842c001",
        "5bf26cbbd43923194854b270",
        "5bf3a82cd439231948877aed",
        "5bf7d63575c26f32dbf7413b",
        "5bfc9d5aec61ca1dd69132a2",
        "5bfd0f32ec61ca1dd69dc77b",
        "5bfe5ae0fe0ea555e6a969ca",
        "5bff3c5cfe0ea555e6bcbf3a",
        "5c062d84a96e33018ff6f0a6",
        "5c0d13b795da9479e12e2ee9",
        "5c1892f726173c3a09ea9aeb",
        "5c1af2e2bee9a723c963d019",
        "5c1b1500bee9a723c96c3e78",
        "5c1dbf200843bc542d8ef8c4",
        "5c1f33f1d33e1f2e4aa6dda4",
        "5c20ca3a0843bc542d94e3e2",
        "5c2b3ed5e611832e8aed46bf",
        "5c34300a73a8df509add216d",
        "5c34529873a8df509ae57b58",
    ]


def val_scans() -> List[str]:
    return [
        "5bb7a08aea1cfa39f1a947ab",
        "5af02e904c8216544b4ab5a2",
        "5be883a4f98cee15019d5b83",
        "5b21e18c58e2823a67a10dd8",
        "5a7d3db14989e929563eb153",
        "5a48ba95c7dab83a7d7b44ed",
        "5be4ab93870d330ff2dce134",
        "5aa0f9d7a9efce63548c69a1",
    ]


def test_scans() -> List[str]:
    return [
        "5b7a3890fc8fcf6781e2593a",
        "5c189f2326173c3a09ed7ef3",
        "5b950c71608de421b1e7318f",
        "5a6400933d809f1d8200af15",
        "59d2657f82ca7774b1ec081d",
        "5ba19a8a360c7c30c1c169df",
        "59817e4a1bd4b175e7038d19",
    ]