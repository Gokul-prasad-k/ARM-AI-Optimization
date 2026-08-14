/**
  ******************************************************************************
  * @file    network_ee_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-14T23:49:04+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_ee_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_ee_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_ee_weights_array_u64[390] = {
  0xbf9402c83f4e5ae3U, 0xc00c1868bf5e211dU, 0xbd664b2d4047a08aU, 0xbf55f8f9be2ae359U,
  0xc0607978bf40fe48U, 0xbfce57743f65424fU, 0xc08182f43f5b9d28U, 0xbf3cf934c08562bfU,
  0xbf8b2b40409f4495U, 0x3e01cf6ec093e5e3U, 0x3f992385bfc6e8f4U, 0x3ecf42d7c03a5cf6U,
  0xbe507b79bfaf8e6cU, 0xc0144a63bfec8883U, 0x3ee000b4bf135f7eU, 0x402e17e1c0acad7dU,
  0x3f6f365e3e437372U, 0xbeaa0af23f0781e7U, 0xbf283f2f40bcd2d0U, 0x3fb7ce7e3fc8414cU,
  0xbf663de840c61872U, 0x3f37af54bfd8f2f4U, 0xbf8597023f16fc79U, 0xbf8759bbc01bda4aU,
  0xbd423bcf3e9636a3U, 0xbeae2b5e3fa75748U, 0x3fc6531f3f8bdd7cU, 0xbf233c9fbf953810U,
  0xbd47e7533f47ef8aU, 0xbf633691bf942159U, 0xbde6932740f8fe32U, 0xbf857d1cbefae1e9U,
  0x403273203f160ed0U, 0xc051720c3eab779eU, 0xc032a010400e92d3U, 0x3f86a51ec004b695U,
  0xbf820a41bfa744d6U, 0x3e0f5d2dbf9de4c4U, 0xbed03e32be80bbedU, 0x3fe32b0dc031660cU,
  0xbc23db383e7cb7a7U, 0xbf956dcec09054a4U, 0xbfd511b4be69e70eU, 0x3f5a8e01bfb6f9d3U,
  0x403bd941408f3a2aU, 0x402ecca83f8a0da8U, 0xbe6b47cc401b5291U, 0x3fb9f548c0eeb2e0U,
  0x3fcf16f7bfdbdc6cU, 0xc02c6b67bf394345U, 0xbf03ebc13ec4c23cU, 0x3ff54f483fa45044U,
  0xc08778bebd7f0698U, 0xc0746fe5bf1d286bU, 0xbfb8f8f8c07b3562U, 0xc0b0212b3eb05866U,
  0x3f8eb6e53fdb89afU, 0xbf94a37e403b3506U, 0xbfaae8b13faadb67U, 0x3f521615bf026a84U,
  0xbeeaffd93f7ffcdcU, 0x3e9c4fb2bf99afadU, 0x3fbbee0f4065f3d5U, 0x3f632d5cbf837e26U,
  0x3e98b0b03fa91fe6U, 0x40be248dbfc13260U, 0x3fc2daf1c001a5e8U, 0xbfb7aff53ef526feU,
  0x40b9f1643fc5a027U, 0x4000a6f63e041d01U, 0x40a188c3bf1b3898U, 0xc000e0d83fd29366U,
  0xbea75780bfbb9685U, 0x3fa38f5f40469218U, 0x3f9b160d3e829c13U, 0xbe77abc940aa082cU,
  0xc0115882bfeb11b0U, 0xc0305218bebe565dU, 0x3effe2fdc01c1e29U, 0x3df17df6c11a7618U,
  0xbef6b411bd26b7ccU, 0x3f9156fbbfa3a754U, 0xbf89de1fc088efacU, 0xc08645abbf80b23bU,
  0x402bbf30c064b8bcU, 0x4079c158c046e997U, 0x3e73498dbf74f333U, 0x3fecbd42bf907f26U,
  0xbfa46efa3f8a557cU, 0x4017cbb2c004b7b5U, 0x3e05fc82bf87dd00U, 0xbdc70d213f969d88U,
  0xbf745a193fc262a1U, 0x3d8578afbff0cbeaU, 0xbf903321c0d1767dU, 0xbfd68afd3fa60207U,
  0x3ec9d5bf3f1b7c23U, 0x3f2b323abff4f648U, 0xbee70725c04fed06U, 0x405659d63fd7ea83U,
  0xbfb7d60bbf9d7b16U, 0xbe9c83413ef9619cU, 0xc087278c3f072bc8U, 0xbea3d3acc04768a3U,
  0xbecb1318be0c87f5U, 0xbf2c6982be973a06U, 0x3e9e237a3ff8df02U, 0x3e2a2120406765daU,
  0x3fd0cc44405ff070U, 0x40650a8b3dfbd2edU, 0xbed40423402d3250U, 0x3f4e70e1c127e0c1U,
  0x3ddb1ed5bf4f8f14U, 0xc087c6bb3f0512b0U, 0xbeb1a11bc0670e1cU, 0x4014010d3ff38ab8U,
  0xc019049b4100fe84U, 0xbee728abbfdc774bU, 0xbfe0b3103f6cd50aU, 0xc069a907bf671079U,
  0xbf1bfbdb3f4befa5U, 0x4022c184bec8a651U, 0xc00765b23ffe9199U, 0x3e23834fbf84b136U,
  0xbfc5c00c3f80bd9fU, 0xbfde418fbfc90039U, 0x3f2080d93f5f5fd6U, 0x3da9b647bf1c345dU,
  0xc038dd51bf9a2d6cU, 0xbfa5b041c022d6b3U, 0xbfad67024013c0cfU, 0x3f35a98d4082c69fU,
  0x4041d50abf241ad5U, 0x3f93d832bfc01f17U, 0x40b66fcabd369abaU, 0x3fb759f23fa5cfd4U,
  0x4033a76cbf42cbc8U, 0x403f2a593f201952U, 0x401155133e87f761U, 0xc060d779c103d459U,
  0x3face6343ebdbab3U, 0xbfea52a93f7c785dU, 0xc00503abbfc2db67U, 0xbfc4f906407ed082U,
  0xbf489147bf6deaf1U, 0x3e65e1e13fa55264U, 0x3f95082cbfa080caU, 0xc0ae5f42bec1f826U,
  0xc0143899c0519dc1U, 0xc09ed32b4086b41cU, 0xbf00440bbf572496U, 0xbf8da1ed3f9cbbb0U,
  0xbfeee517be3ad612U, 0xbca8ccac3f95a397U, 0xbf871a2cbfe9e82dU, 0x407a699cbfb7bbf7U,
  0xc00f5d65bf816333U, 0xbf4ada773e4f6cd3U, 0x4004a8cbc0b6e898U, 0xbf6df5453f1eff63U,
  0xc060c5e1c041a988U, 0xc02bab06bec1e56dU, 0x3f2984d44012f1d2U, 0xbece59ee3fe66be9U,
  0xc080088dc0001d7cU, 0x402dafcdbf87d320U, 0x4056eedc3e389fafU, 0x3fc0d62e400437d0U,
  0xbda294a740cc97a9U, 0xc090afa4bfec1f78U, 0x3f9976933ed9b83cU, 0xbfc1cb28c11b4b55U,
  0xbdc450febfd2e5b1U, 0xbffc406540167f70U, 0xbfad7edc3f61fb26U, 0xbf9d2c5640e91f70U,
  0x3f29247bbf29d480U, 0x3f4904054018dabaU, 0xbd59650340562a02U, 0xbfa08e88c0881b3cU,
  0x3f66087ac051be5eU, 0xc0b124264062b97fU, 0xbf060641400dc41cU, 0x4073f0e3bf269899U,
  0xbf32aab3bfa28bd0U, 0xbfc7366c3fb78746U, 0x3b10aeb3c04d2028U, 0x3e77bd14c048b3c9U,
  0x402bcab0c00567baU, 0xbeded82ebf066d90U, 0x3fd5c7c2beee13e4U, 0xbf978933bf43db51U,
  0x4008f0f53fe4a6a2U, 0x3fe4624d3fce2f4dU, 0x3f03c8dd3f22ddbeU, 0xbfb13b91bfb48409U,
  0xbfb8f4f5c00e791dU, 0xbea1b22b3fad2958U, 0x40becc79bfd8b55aU, 0xbfd7c2cd3f340732U,
  0xbe80a8b0bffdeb39U, 0xc031bd5d3e8b06fdU, 0xc0741e66406083b5U, 0x3f9c5b114047e8e6U,
  0xbfc449adbf369dd1U, 0xc08396a4400bf662U, 0x3f10838440464db1U, 0xc05f8dffc035e133U,
  0xc01156793f23fec3U, 0x3fc86431c02829feU, 0xbfde8296bf54f418U, 0xc02388f8c0f9cc3dU,
  0x400bbcafc0e630b9U, 0xbf8b1a6ac0523dbaU, 0x3fec05a1bfd6eef2U, 0xbf1aedf93f87ae51U,
  0xbef88cafc08402fdU, 0x40c65d0b3f92a35bU, 0x406b48a7bfddad96U, 0xbf6aea6f3f9b6b91U,
  0x403b4f1ebea42086U, 0x4006a0ae3f4cfe6dU, 0x3e898befc098a6d0U, 0x3f3514303e8de1ceU,
  0xc022ec41bfedc5f1U, 0x3fcff50ec05b3012U, 0xc000e3c9403ca531U, 0xbfb5933ac0b55edfU,
  0x4047effdbf12ae9bU, 0xbef5879f3e7dc902U, 0xbf3d24623f15f982U, 0x3f8bc22b410d17cfU,
  0x3e47fe50bff2aedcU, 0x3e43c49fbedcf79cU, 0xc0314d7abffd9230U, 0x3f20748cc03bfff5U,
  0x4018a8c6bf99ee30U, 0x3f501eb5c0de61d1U, 0xbe86bb9dc03fabe0U, 0x3ff0fe2a3f37bd9dU,
  0xbdd28470bef823a5U, 0x3faf493c3fd99981U, 0xbe9c11e1401c0f94U, 0xbe7a4d0e3fc719abU,
  0xbf97b84d3fd15799U, 0x40371a4f409d5d62U, 0x4004d901bf0a2ed1U, 0xc0603f1d3f934c59U,
  0x3dc5891dbee86cffU, 0xc0153fce3f5a9c76U, 0xbb7657133f0b4d0bU, 0xbfa5f14e3fb80f8cU,
  0xbfcce0fec04a3e7aU, 0x3f7f1564c01b0cb5U, 0x3faedf4640aeed56U, 0xc0361a3e3e9e82a9U,
  0x3f4f428bbe341654U, 0xbf5fc65a3dea04dbU, 0xbe1cdc2e4039795aU, 0xc00bbd02bfbb3ddbU,
  0x3f5c0e2f3f8c6e17U, 0xbf07a73bbf5cc473U, 0x40805e273f2db691U, 0xbdc4a5ea40e6721aU,
  0x3f75df90bfa2fa67U, 0x3fe1726ac00c2dafU, 0x3e6deb7cc05253b6U, 0x3f370d15c09048c0U,
  0x402e3111404e0d4eU, 0x3fd39071c065f234U, 0x3f8a6ca9c005b4a0U, 0xbf393929bf3e3136U,
  0x3f473536bf17befbU, 0xbf663af83f9af0ecU, 0x3f2e5660bef325f3U, 0xc016b142be9cdec0U,
  0xc018cc57bfe71cc6U, 0xbf010544406031fdU, 0x3f0d46edbff22f52U, 0xbfbfe430bf8c9072U,
  0x3fbe8aadc0518cc7U, 0xc02132dd3d3d31efU, 0x3f2f60f53dfb30d0U, 0x3edf383b3ff34caeU,
  0x4063382ac0808bbcU, 0xbe101c8bc0181727U, 0xbfa21987c0c2e666U, 0xc02722bb3f6a9c21U,
  0xbf0840a83e382ee3U, 0x404a2b6abf2d0e9aU, 0x4027f55bc07f2e38U, 0x3fd1294640599f12U,
  0x408b79ea40207274U, 0x3e0f9bbdbf3db692U, 0xc08f99313f90b4a5U, 0xbfc437bcc0b6fd83U,
  0xbd708e7cc030790bU, 0x403e7c513f75a6ebU, 0x404249a3c06a1950U, 0x3f84a1414083b15aU,
  0xc0374149c05272c2U, 0x3f91056bc0123902U, 0x3f885f0bc051bca8U, 0x3ff26e9dc0e2e07aU,
  0x3eb82126400fd862U, 0xbfce1852bf85d394U, 0x3f6b6058c063da1aU, 0xbf64d87f40cfab3fU,
  0x3f6f7fc94064f2eeU, 0x40ae135cc09a39daU, 0xbfccd360404d9aa0U, 0xbfad969f3f05da2dU,
  0xc031febdbdb98855U, 0x3fc5113ac0129294U, 0xc04a2c853c0be831U, 0x3dea1d9dbe62ed06U,
  0xc07f6f14405b88f8U, 0xbff6cf6e3de68efeU, 0x3f05ace8c0417e85U, 0xbfaa0d6f3f688ba4U,
  0x3e0a0f183fcf4d30U, 0xbfc04c423e8aa224U, 0xbff748523f8caa10U, 0x3f03cc14bf23b4a2U,
  0x3fab2306bd2ca91aU, 0xbeb85793bfc7d53aU, 0xbfab588d3f3cb9b7U, 0x3f0484e3c00fe9fbU,
  0x3ec977f03f44683fU, 0xc1116e43c028bb16U, 0x4064966a3ea1ec24U, 0xbfcedffcc0888ef6U,
  0xc0a8b124bf706ee2U, 0xbef6d8a53ed44a5eU, 0x3fee6e55c073d965U, 0x3f839862bf363fd4U,
  0xc0b587b43f3cfe11U, 0xc0725cedc01c94e5U, 0xc0568ef13f655925U, 0xc133f3183fa72662U,
  0x3f4df8a23db87d5cU, 0x402bcf373ebd2b00U, 0xbf1500edbec5e045U, 0x3f51d4c54041333cU,
  0xbc20c45f3f9cbe82U, 0x3fa90540c0e99b41U, 0xc01163a3bf918154U, 0xbe3744a4bf0197acU,
  0xc0da9260401ba17cU, 0xbd546729404b6352U, 0xc097066b3f295b46U, 0x3f1d2d2abe71563fU,
  0x3f99d987be81e2feU, 0xbfbeba4d4052beaaU, 0x3f106eebc08c3526U, 0xbd32ff65bf93a1b9U,
  0xc05bb0da3f98211cU, 0xbf9ca6d53f1ef075U, 0xc0ca8c98bf5b6ab7U, 0xbc839ce7c09b8680U,
  0xbf6fbaabc020ed1dU, 0xbdb175db4065db1eU, 0xbf794eed3ec976b1U, 0x3db6390d404afb8dU,
  0xc00c78f5bfd0cad5U, 0x3e53f07b400c75ecU, 0xbd5b34a53f708d84U, 0xbfef928040ef1b9fU,
  0xbeb505ca3f1278c9U, 0x3ff27cdfbf87368cU, 0x3f505dc3bf0a54c7U, 0x40633477bf9d6b8bU,
  0x3fe73313c0764074U, 0xbe03b384c0086b81U, 0x3f0f73223f22b9efU, 0x403cbf783da6721fU,
  0x3fd322b73fab482dU, 0xc02bd759c00688a1U, 0xbe95597f3f7470aaU, 0xbf91d7aa3e8e4140U,
  0xbe4d296c3daa3f45U, 0x3f460f08404d1bd3U, 0xbfd05ba2bf76b1a7U, 0x40556868be3a650eU,
  0x3e8dc258bf2820dfU, 0x3d149a06bea40334U, 0xbf389c32be2b828cU, 0xbf17d225bf038777U,
  0xbf29540d3ea1ac2fU, 0x3f6cb91b3eda226aU,
};


ai_handle g_network_ee_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_ee_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

