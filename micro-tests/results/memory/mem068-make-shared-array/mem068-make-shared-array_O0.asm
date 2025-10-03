
/Users/jim/work/cppfort/micro-tests/results/memory/mem068-make-shared-array/mem068-make-shared-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z22test_make_shared_arrayv>:
100000538: d10143ff    	sub	sp, sp, #0x50
10000053c: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000540: 910103fd    	add	x29, sp, #0x40
100000544: d10043a8    	sub	x8, x29, #0x10
100000548: f9000be8    	str	x8, [sp, #0x10]
10000054c: d28000a0    	mov	x0, #0x5                ; =5
100000550: 9400001f    	bl	0x1000005cc <__ZNSt3__111make_sharedB8ne200100IA_iLi0EEENS_10shared_ptrIT_EEm>
100000554: f9400be0    	ldr	x0, [sp, #0x10]
100000558: d2800041    	mov	x1, #0x2                ; =2
10000055c: 9400002c    	bl	0x10000060c <__ZNKSt3__110shared_ptrIA_iEixB8ne200100El>
100000560: f9000fe0    	str	x0, [sp, #0x18]
100000564: 14000001    	b	0x100000568 <__Z22test_make_shared_arrayv+0x30>
100000568: f9400fe9    	ldr	x9, [sp, #0x18]
10000056c: 52800548    	mov	w8, #0x2a               ; =42
100000570: b9000128    	str	w8, [x9]
100000574: d10043a0    	sub	x0, x29, #0x10
100000578: d2800041    	mov	x1, #0x2                ; =2
10000057c: 94000024    	bl	0x10000060c <__ZNKSt3__110shared_ptrIA_iEixB8ne200100El>
100000580: f90007e0    	str	x0, [sp, #0x8]
100000584: 14000001    	b	0x100000588 <__Z22test_make_shared_arrayv+0x50>
100000588: f94007e8    	ldr	x8, [sp, #0x8]
10000058c: b9400108    	ldr	w8, [x8]
100000590: b90007e8    	str	w8, [sp, #0x4]
100000594: d10043a0    	sub	x0, x29, #0x10
100000598: 94000026    	bl	0x100000630 <__ZNSt3__110shared_ptrIA_iED1B8ne200100Ev>
10000059c: b94007e0    	ldr	w0, [sp, #0x4]
1000005a0: a9447bfd    	ldp	x29, x30, [sp, #0x40]
1000005a4: 910143ff    	add	sp, sp, #0x50
1000005a8: d65f03c0    	ret
1000005ac: f81e83a0    	stur	x0, [x29, #-0x18]
1000005b0: aa0103e8    	mov	x8, x1
1000005b4: b81e43a8    	stur	w8, [x29, #-0x1c]
1000005b8: d10043a0    	sub	x0, x29, #0x10
1000005bc: 9400001d    	bl	0x100000630 <__ZNSt3__110shared_ptrIA_iED1B8ne200100Ev>
1000005c0: 14000001    	b	0x1000005c4 <__Z22test_make_shared_arrayv+0x8c>
1000005c4: f85e83a0    	ldur	x0, [x29, #-0x18]
1000005c8: 9400042e    	bl	0x100001680 <___stack_chk_guard+0x100001680>

00000001000005cc <__ZNSt3__111make_sharedB8ne200100IA_iLi0EEENS_10shared_ptrIT_EEm>:
1000005cc: d10103ff    	sub	sp, sp, #0x40
1000005d0: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000005d4: 9100c3fd    	add	x29, sp, #0x30
1000005d8: f9000be8    	str	x8, [sp, #0x10]
1000005dc: f81f83a8    	stur	x8, [x29, #-0x8]
1000005e0: f81f03a0    	stur	x0, [x29, #-0x10]
1000005e4: d10047a0    	sub	x0, x29, #0x11
1000005e8: f90007e0    	str	x0, [sp, #0x8]
1000005ec: 94000060    	bl	0x10000076c <__ZNSt3__19allocatorIA_iEC1B8ne200100Ev>
1000005f0: f94007e0    	ldr	x0, [sp, #0x8]
1000005f4: f9400be8    	ldr	x8, [sp, #0x10]
1000005f8: f85f03a1    	ldur	x1, [x29, #-0x10]
1000005fc: 94000020    	bl	0x10000067c <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_>
100000600: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000604: 910103ff    	add	sp, sp, #0x40
100000608: d65f03c0    	ret

000000010000060c <__ZNKSt3__110shared_ptrIA_iEixB8ne200100El>:
10000060c: d10043ff    	sub	sp, sp, #0x10
100000610: f90007e0    	str	x0, [sp, #0x8]
100000614: f90003e1    	str	x1, [sp]
100000618: f94007e8    	ldr	x8, [sp, #0x8]
10000061c: f9400108    	ldr	x8, [x8]
100000620: f94003e9    	ldr	x9, [sp]
100000624: 8b090900    	add	x0, x8, x9, lsl #2
100000628: 910043ff    	add	sp, sp, #0x10
10000062c: d65f03c0    	ret

0000000100000630 <__ZNSt3__110shared_ptrIA_iED1B8ne200100Ev>:
100000630: d10083ff    	sub	sp, sp, #0x20
100000634: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000638: 910043fd    	add	x29, sp, #0x10
10000063c: f90007e0    	str	x0, [sp, #0x8]
100000640: f94007e0    	ldr	x0, [sp, #0x8]
100000644: f90003e0    	str	x0, [sp]
100000648: 940003c2    	bl	0x100001550 <__ZNSt3__110shared_ptrIA_iED2B8ne200100Ev>
10000064c: f94003e0    	ldr	x0, [sp]
100000650: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000654: 910083ff    	add	sp, sp, #0x20
100000658: d65f03c0    	ret

000000010000065c <_main>:
10000065c: d10083ff    	sub	sp, sp, #0x20
100000660: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000664: 910043fd    	add	x29, sp, #0x10
100000668: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000066c: 97ffffb3    	bl	0x100000538 <__Z22test_make_shared_arrayv>
100000670: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000674: 910083ff    	add	sp, sp, #0x20
100000678: d65f03c0    	ret

000000010000067c <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_>:
10000067c: d10203ff    	sub	sp, sp, #0x80
100000680: a9077bfd    	stp	x29, x30, [sp, #0x70]
100000684: 9101c3fd    	add	x29, sp, #0x70
100000688: f9000be8    	str	x8, [sp, #0x10]
10000068c: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000690: f9400929    	ldr	x9, [x9, #0x10]
100000694: f9400129    	ldr	x9, [x9]
100000698: f81f83a9    	stur	x9, [x29, #-0x8]
10000069c: f81d83a8    	stur	x8, [x29, #-0x28]
1000006a0: f81d03a0    	stur	x0, [x29, #-0x30]
1000006a4: f9001fe1    	str	x1, [sp, #0x38]
1000006a8: f9401fe0    	ldr	x0, [sp, #0x38]
1000006ac: 9400003b    	bl	0x100000798 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em>
1000006b0: d2800108    	mov	x8, #0x8                ; =8
1000006b4: 9ac80801    	udiv	x1, x0, x8
1000006b8: d10083a0    	sub	x0, x29, #0x20
1000006bc: 9400004f    	bl	0x1000007f8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEEC1B8ne200100INS1_IA_iEEEET_m>
1000006c0: 14000001    	b	0x1000006c4 <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0x48>
1000006c4: d10083a0    	sub	x0, x29, #0x20
1000006c8: 94000059    	bl	0x10000082c <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE5__getB8ne200100Ev>
1000006cc: f90017e0    	str	x0, [sp, #0x28]
1000006d0: f94017e0    	ldr	x0, [sp, #0x28]
1000006d4: f85d03a1    	ldur	x1, [x29, #-0x30]
1000006d8: 9100e3e2    	add	x2, sp, #0x38
1000006dc: 9400005a    	bl	0x100000844 <__ZNSt3__114__construct_atB8ne200100INS_31__unbounded_array_control_blockIA_iNS_9allocatorIS2_EEEEJRKS4_RmEPS5_EEPT_SB_DpOT0_>
1000006e0: 14000001    	b	0x1000006e4 <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0x68>
1000006e4: d10083a0    	sub	x0, x29, #0x20
1000006e8: f90007e0    	str	x0, [sp, #0x8]
1000006ec: 94000063    	bl	0x100000878 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE13__release_ptrB8ne200100Ev>
1000006f0: f94017e0    	ldr	x0, [sp, #0x28]
1000006f4: 94000093    	bl	0x100000940 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE10__get_dataB8ne200100Ev>
1000006f8: f9400be8    	ldr	x8, [sp, #0x10]
1000006fc: f94017e1    	ldr	x1, [sp, #0x28]
100000700: 940003e3    	bl	0x10000168c <___stack_chk_guard+0x10000168c>
100000704: f94007e0    	ldr	x0, [sp, #0x8]
100000708: 94000094    	bl	0x100000958 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEED1B8ne200100Ev>
10000070c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000710: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000714: f9400908    	ldr	x8, [x8, #0x10]
100000718: f9400108    	ldr	x8, [x8]
10000071c: eb090108    	subs	x8, x8, x9
100000720: 54000060    	b.eq	0x10000072c <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0xb0>
100000724: 14000001    	b	0x100000728 <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0xac>
100000728: 940003dc    	bl	0x100001698 <___stack_chk_guard+0x100001698>
10000072c: a9477bfd    	ldp	x29, x30, [sp, #0x70]
100000730: 910203ff    	add	sp, sp, #0x80
100000734: d65f03c0    	ret
100000738: f90013e0    	str	x0, [sp, #0x20]
10000073c: aa0103e8    	mov	x8, x1
100000740: b9001fe8    	str	w8, [sp, #0x1c]
100000744: d10083a0    	sub	x0, x29, #0x20
100000748: 94000084    	bl	0x100000958 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEED1B8ne200100Ev>
10000074c: 14000001    	b	0x100000750 <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0xd4>
100000750: f94013e0    	ldr	x0, [sp, #0x20]
100000754: f90003e0    	str	x0, [sp]
100000758: 14000003    	b	0x100000764 <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0xe8>
10000075c: f90003e0    	str	x0, [sp]
100000760: 14000001    	b	0x100000764 <__ZNSt3__133__allocate_shared_unbounded_arrayB8ne200100IA_iNS_9allocatorIS1_EEJEEENS_10shared_ptrIT_EERKT0_mDpOT1_+0xe8>
100000764: f94003e0    	ldr	x0, [sp]
100000768: 940003c6    	bl	0x100001680 <___stack_chk_guard+0x100001680>

000000010000076c <__ZNSt3__19allocatorIA_iEC1B8ne200100Ev>:
10000076c: d10083ff    	sub	sp, sp, #0x20
100000770: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000774: 910043fd    	add	x29, sp, #0x10
100000778: f90007e0    	str	x0, [sp, #0x8]
10000077c: f94007e0    	ldr	x0, [sp, #0x8]
100000780: f90003e0    	str	x0, [sp]
100000784: 94000363    	bl	0x100001510 <__ZNSt3__19allocatorIA_iEC2B8ne200100Ev>
100000788: f94003e0    	ldr	x0, [sp]
10000078c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000790: 910083ff    	add	sp, sp, #0x20
100000794: d65f03c0    	ret

0000000100000798 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em>:
100000798: d10083ff    	sub	sp, sp, #0x20
10000079c: f9000fe0    	str	x0, [sp, #0x18]
1000007a0: f9400fe8    	ldr	x8, [sp, #0x18]
1000007a4: b50000a8    	cbnz	x8, 0x1000007b8 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em+0x20>
1000007a8: 14000001    	b	0x1000007ac <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em+0x14>
1000007ac: d2800508    	mov	x8, #0x28               ; =40
1000007b0: f90003e8    	str	x8, [sp]
1000007b4: 14000007    	b	0x1000007d0 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em+0x38>
1000007b8: f9400fe8    	ldr	x8, [sp, #0x18]
1000007bc: f1000508    	subs	x8, x8, #0x1
1000007c0: d37ef508    	lsl	x8, x8, #2
1000007c4: 9100a108    	add	x8, x8, #0x28
1000007c8: f90003e8    	str	x8, [sp]
1000007cc: 14000001    	b	0x1000007d0 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em+0x38>
1000007d0: f94003e8    	ldr	x8, [sp]
1000007d4: f9000be8    	str	x8, [sp, #0x10]
1000007d8: d2800088    	mov	x8, #0x4                ; =4
1000007dc: f90007e8    	str	x8, [sp, #0x8]
1000007e0: f9400be8    	ldr	x8, [sp, #0x10]
1000007e4: 91001108    	add	x8, x8, #0x4
1000007e8: f1000508    	subs	x8, x8, #0x1
1000007ec: 927ef500    	and	x0, x8, #0xfffffffffffffffc
1000007f0: 910083ff    	add	sp, sp, #0x20
1000007f4: d65f03c0    	ret

00000001000007f8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEEC1B8ne200100INS1_IA_iEEEET_m>:
1000007f8: d100c3ff    	sub	sp, sp, #0x30
1000007fc: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000800: 910083fd    	add	x29, sp, #0x20
100000804: f9000be0    	str	x0, [sp, #0x10]
100000808: f90007e1    	str	x1, [sp, #0x8]
10000080c: f9400be0    	ldr	x0, [sp, #0x10]
100000810: f90003e0    	str	x0, [sp]
100000814: f94007e1    	ldr	x1, [sp, #0x8]
100000818: 9400005b    	bl	0x100000984 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEEC2B8ne200100INS1_IA_iEEEET_m>
10000081c: f94003e0    	ldr	x0, [sp]
100000820: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000824: 9100c3ff    	add	sp, sp, #0x30
100000828: d65f03c0    	ret

000000010000082c <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE5__getB8ne200100Ev>:
10000082c: d10043ff    	sub	sp, sp, #0x10
100000830: f90007e0    	str	x0, [sp, #0x8]
100000834: f94007e8    	ldr	x8, [sp, #0x8]
100000838: f9400900    	ldr	x0, [x8, #0x10]
10000083c: 910043ff    	add	sp, sp, #0x10
100000840: d65f03c0    	ret

0000000100000844 <__ZNSt3__114__construct_atB8ne200100INS_31__unbounded_array_control_blockIA_iNS_9allocatorIS2_EEEEJRKS4_RmEPS5_EEPT_SB_DpOT0_>:
100000844: d100c3ff    	sub	sp, sp, #0x30
100000848: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000084c: 910083fd    	add	x29, sp, #0x20
100000850: f81f83a0    	stur	x0, [x29, #-0x8]
100000854: f9000be1    	str	x1, [sp, #0x10]
100000858: f90007e2    	str	x2, [sp, #0x8]
10000085c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000860: f9400be1    	ldr	x1, [sp, #0x10]
100000864: f94007e2    	ldr	x2, [sp, #0x8]
100000868: 940000ec    	bl	0x100000c18 <__ZNSt3__112construct_atB8ne200100INS_31__unbounded_array_control_blockIA_iNS_9allocatorIS2_EEEEJRKS4_RmEPS5_EEPT_SB_DpOT0_>
10000086c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000870: 9100c3ff    	add	sp, sp, #0x30
100000874: d65f03c0    	ret

0000000100000878 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE13__release_ptrB8ne200100Ev>:
100000878: d10043ff    	sub	sp, sp, #0x10
10000087c: f90007e0    	str	x0, [sp, #0x8]
100000880: f94007e8    	ldr	x8, [sp, #0x8]
100000884: f9400909    	ldr	x9, [x8, #0x10]
100000888: f90003e9    	str	x9, [sp]
10000088c: f900091f    	str	xzr, [x8, #0x10]
100000890: f94003e0    	ldr	x0, [sp]
100000894: 910043ff    	add	sp, sp, #0x10
100000898: d65f03c0    	ret

000000010000089c <__ZNSt3__110shared_ptrIA_iE27__create_with_control_blockB8ne200100IiNS_31__unbounded_array_control_blockIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_>:
10000089c: d10143ff    	sub	sp, sp, #0x50
1000008a0: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000008a4: 910103fd    	add	x29, sp, #0x40
1000008a8: f9000fe8    	str	x8, [sp, #0x18]
1000008ac: aa0003e8    	mov	x8, x0
1000008b0: f9400fe0    	ldr	x0, [sp, #0x18]
1000008b4: aa0003e9    	mov	x9, x0
1000008b8: f81f83a9    	stur	x9, [x29, #-0x8]
1000008bc: f81f03a8    	stur	x8, [x29, #-0x10]
1000008c0: f81e83a1    	stur	x1, [x29, #-0x18]
1000008c4: 52800008    	mov	w8, #0x0                ; =0
1000008c8: 52800029    	mov	w9, #0x1                ; =1
1000008cc: b90023e9    	str	w9, [sp, #0x20]
1000008d0: 12000108    	and	w8, w8, #0x1
1000008d4: 12000108    	and	w8, w8, #0x1
1000008d8: 381e73a8    	sturb	w8, [x29, #-0x19]
1000008dc: 940002db    	bl	0x100001448 <__ZNSt3__110shared_ptrIA_iEC1B8ne200100Ev>
1000008e0: f9400fe0    	ldr	x0, [sp, #0x18]
1000008e4: f85f03a8    	ldur	x8, [x29, #-0x10]
1000008e8: f9000008    	str	x8, [x0]
1000008ec: f85e83a8    	ldur	x8, [x29, #-0x18]
1000008f0: f9000408    	str	x8, [x0, #0x8]
1000008f4: f940000a    	ldr	x10, [x0]
1000008f8: f9400008    	ldr	x8, [x0]
1000008fc: 910003e9    	mov	x9, sp
100000900: f900012a    	str	x10, [x9]
100000904: f9000528    	str	x8, [x9, #0x8]
100000908: 940002db    	bl	0x100001474 <__ZNSt3__110shared_ptrIA_iE18__enable_weak_thisB8ne200100Ez>
10000090c: b94023e9    	ldr	w9, [sp, #0x20]
100000910: 12000128    	and	w8, w9, #0x1
100000914: 0a090108    	and	w8, w8, w9
100000918: 381e73a8    	sturb	w8, [x29, #-0x19]
10000091c: 385e73a8    	ldurb	w8, [x29, #-0x19]
100000920: 370000a8    	tbnz	w8, #0x0, 0x100000934 <__ZNSt3__110shared_ptrIA_iE27__create_with_control_blockB8ne200100IiNS_31__unbounded_array_control_blockIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_+0x98>
100000924: 14000001    	b	0x100000928 <__ZNSt3__110shared_ptrIA_iE27__create_with_control_blockB8ne200100IiNS_31__unbounded_array_control_blockIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_+0x8c>
100000928: f9400fe0    	ldr	x0, [sp, #0x18]
10000092c: 97ffff41    	bl	0x100000630 <__ZNSt3__110shared_ptrIA_iED1B8ne200100Ev>
100000930: 14000001    	b	0x100000934 <__ZNSt3__110shared_ptrIA_iE27__create_with_control_blockB8ne200100IiNS_31__unbounded_array_control_blockIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_+0x98>
100000934: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000938: 910143ff    	add	sp, sp, #0x50
10000093c: d65f03c0    	ret

0000000100000940 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE10__get_dataB8ne200100Ev>:
100000940: d10043ff    	sub	sp, sp, #0x10
100000944: f90007e0    	str	x0, [sp, #0x8]
100000948: f94007e8    	ldr	x8, [sp, #0x8]
10000094c: 91008100    	add	x0, x8, #0x20
100000950: 910043ff    	add	sp, sp, #0x10
100000954: d65f03c0    	ret

0000000100000958 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEED1B8ne200100Ev>:
100000958: d10083ff    	sub	sp, sp, #0x20
10000095c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000960: 910043fd    	add	x29, sp, #0x10
100000964: f90007e0    	str	x0, [sp, #0x8]
100000968: f94007e0    	ldr	x0, [sp, #0x8]
10000096c: f90003e0    	str	x0, [sp]
100000970: 940002cc    	bl	0x1000014a0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEED2B8ne200100Ev>
100000974: f94003e0    	ldr	x0, [sp]
100000978: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000097c: 910083ff    	add	sp, sp, #0x20
100000980: d65f03c0    	ret

0000000100000984 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEEC2B8ne200100INS1_IA_iEEEET_m>:
100000984: d100c3ff    	sub	sp, sp, #0x30
100000988: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000098c: 910083fd    	add	x29, sp, #0x20
100000990: f9000be0    	str	x0, [sp, #0x10]
100000994: f90007e1    	str	x1, [sp, #0x8]
100000998: f9400be0    	ldr	x0, [sp, #0x10]
10000099c: f90003e0    	str	x0, [sp]
1000009a0: d10007a1    	sub	x1, x29, #0x1
1000009a4: 9400000c    	bl	0x1000009d4 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEEC1B8ne200100IA_iEERKNS0_IT_EE>
1000009a8: f94003e0    	ldr	x0, [sp]
1000009ac: f94007e8    	ldr	x8, [sp, #0x8]
1000009b0: f9000408    	str	x8, [x0, #0x8]
1000009b4: f9400401    	ldr	x1, [x0, #0x8]
1000009b8: 94000014    	bl	0x100000a08 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE8allocateB8ne200100ERS4_m>
1000009bc: aa0003e8    	mov	x8, x0
1000009c0: f94003e0    	ldr	x0, [sp]
1000009c4: f9000808    	str	x8, [x0, #0x10]
1000009c8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000009cc: 9100c3ff    	add	sp, sp, #0x30
1000009d0: d65f03c0    	ret

00000001000009d4 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEEC1B8ne200100IA_iEERKNS0_IT_EE>:
1000009d4: d100c3ff    	sub	sp, sp, #0x30
1000009d8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000009dc: 910083fd    	add	x29, sp, #0x20
1000009e0: f81f83a0    	stur	x0, [x29, #-0x8]
1000009e4: f9000be1    	str	x1, [sp, #0x10]
1000009e8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000009ec: f90007e0    	str	x0, [sp, #0x8]
1000009f0: f9400be1    	ldr	x1, [sp, #0x10]
1000009f4: 94000010    	bl	0x100000a34 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEEC2B8ne200100IA_iEERKNS0_IT_EE>
1000009f8: f94007e0    	ldr	x0, [sp, #0x8]
1000009fc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a00: 9100c3ff    	add	sp, sp, #0x30
100000a04: d65f03c0    	ret

0000000100000a08 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE8allocateB8ne200100ERS4_m>:
100000a08: d10083ff    	sub	sp, sp, #0x20
100000a0c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a10: 910043fd    	add	x29, sp, #0x10
100000a14: f90007e0    	str	x0, [sp, #0x8]
100000a18: f90003e1    	str	x1, [sp]
100000a1c: f94007e0    	ldr	x0, [sp, #0x8]
100000a20: f94003e1    	ldr	x1, [sp]
100000a24: 94000015    	bl	0x100000a78 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEE8allocateB8ne200100Em>
100000a28: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a2c: 910083ff    	add	sp, sp, #0x20
100000a30: d65f03c0    	ret

0000000100000a34 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEEC2B8ne200100IA_iEERKNS0_IT_EE>:
100000a34: d100c3ff    	sub	sp, sp, #0x30
100000a38: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a3c: 910083fd    	add	x29, sp, #0x20
100000a40: f81f83a0    	stur	x0, [x29, #-0x8]
100000a44: f9000be1    	str	x1, [sp, #0x10]
100000a48: f85f83a0    	ldur	x0, [x29, #-0x8]
100000a4c: f90007e0    	str	x0, [sp, #0x8]
100000a50: 94000005    	bl	0x100000a64 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__sp_aligned_storageILm8EEEEEEC2B8ne200100Ev>
100000a54: f94007e0    	ldr	x0, [sp, #0x8]
100000a58: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a5c: 9100c3ff    	add	sp, sp, #0x30
100000a60: d65f03c0    	ret

0000000100000a64 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__sp_aligned_storageILm8EEEEEEC2B8ne200100Ev>:
100000a64: d10043ff    	sub	sp, sp, #0x10
100000a68: f90007e0    	str	x0, [sp, #0x8]
100000a6c: f94007e0    	ldr	x0, [sp, #0x8]
100000a70: 910043ff    	add	sp, sp, #0x10
100000a74: d65f03c0    	ret

0000000100000a78 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEE8allocateB8ne200100Em>:
100000a78: d100c3ff    	sub	sp, sp, #0x30
100000a7c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a80: 910083fd    	add	x29, sp, #0x20
100000a84: f81f83a0    	stur	x0, [x29, #-0x8]
100000a88: f9000be1    	str	x1, [sp, #0x10]
100000a8c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000a90: f9400be8    	ldr	x8, [sp, #0x10]
100000a94: f90007e8    	str	x8, [sp, #0x8]
100000a98: 94000303    	bl	0x1000016a4 <___stack_chk_guard+0x1000016a4>
100000a9c: f94007e8    	ldr	x8, [sp, #0x8]
100000aa0: eb000108    	subs	x8, x8, x0
100000aa4: 54000069    	b.ls	0x100000ab0 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEE8allocateB8ne200100Em+0x38>
100000aa8: 14000001    	b	0x100000aac <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEE8allocateB8ne200100Em+0x34>
100000aac: 94000011    	bl	0x100000af0 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
100000ab0: f9400be0    	ldr	x0, [sp, #0x10]
100000ab4: d2800101    	mov	x1, #0x8                ; =8
100000ab8: 9400001b    	bl	0x100000b24 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEPT_NS_15__element_countEm>
100000abc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ac0: 9100c3ff    	add	sp, sp, #0x30
100000ac4: d65f03c0    	ret

0000000100000ac8 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE8max_sizeB8ne200100IS4_vLi0EEEmRKS4_>:
100000ac8: d10083ff    	sub	sp, sp, #0x20
100000acc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ad0: 910043fd    	add	x29, sp, #0x10
100000ad4: f90007e0    	str	x0, [sp, #0x8]
100000ad8: 9400002e    	bl	0x100000b90 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000adc: d2800108    	mov	x8, #0x8                ; =8
100000ae0: 9ac80800    	udiv	x0, x0, x8
100000ae4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ae8: 910083ff    	add	sp, sp, #0x20
100000aec: d65f03c0    	ret

0000000100000af0 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000af0: d10083ff    	sub	sp, sp, #0x20
100000af4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000af8: 910043fd    	add	x29, sp, #0x10
100000afc: d2800100    	mov	x0, #0x8                ; =8
100000b00: 940002ec    	bl	0x1000016b0 <___stack_chk_guard+0x1000016b0>
100000b04: f90007e0    	str	x0, [sp, #0x8]
100000b08: 940002ed    	bl	0x1000016bc <___stack_chk_guard+0x1000016bc>
100000b0c: f94007e0    	ldr	x0, [sp, #0x8]
100000b10: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000b14: f9402021    	ldr	x1, [x1, #0x40]
100000b18: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000b1c: f9402442    	ldr	x2, [x2, #0x48]
100000b20: 940002ea    	bl	0x1000016c8 <___stack_chk_guard+0x1000016c8>

0000000100000b24 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEPT_NS_15__element_countEm>:
100000b24: d10103ff    	sub	sp, sp, #0x40
100000b28: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000b2c: 9100c3fd    	add	x29, sp, #0x30
100000b30: f81f03a0    	stur	x0, [x29, #-0x10]
100000b34: f9000fe1    	str	x1, [sp, #0x18]
100000b38: f85f03a8    	ldur	x8, [x29, #-0x10]
100000b3c: d37df108    	lsl	x8, x8, #3
100000b40: f9000be8    	str	x8, [sp, #0x10]
100000b44: f9400fe0    	ldr	x0, [sp, #0x18]
100000b48: 94000019    	bl	0x100000bac <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000b4c: 36000120    	tbz	w0, #0x0, 0x100000b70 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEPT_NS_15__element_countEm+0x4c>
100000b50: 14000001    	b	0x100000b54 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEPT_NS_15__element_countEm+0x30>
100000b54: f9400fe8    	ldr	x8, [sp, #0x18]
100000b58: f90007e8    	str	x8, [sp, #0x8]
100000b5c: f9400be0    	ldr	x0, [sp, #0x10]
100000b60: f94007e1    	ldr	x1, [sp, #0x8]
100000b64: 94000019    	bl	0x100000bc8 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__sp_aligned_storageILm8EEEJmSt11align_val_tEEEPvDpT0_>
100000b68: f81f83a0    	stur	x0, [x29, #-0x8]
100000b6c: 14000005    	b	0x100000b80 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEPT_NS_15__element_countEm+0x5c>
100000b70: f9400be0    	ldr	x0, [sp, #0x10]
100000b74: 94000020    	bl	0x100000bf4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__sp_aligned_storageILm8EEEEEPvm>
100000b78: f81f83a0    	stur	x0, [x29, #-0x8]
100000b7c: 14000001    	b	0x100000b80 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEPT_NS_15__element_countEm+0x5c>
100000b80: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b84: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000b88: 910103ff    	add	sp, sp, #0x40
100000b8c: d65f03c0    	ret

0000000100000b90 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000b90: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000b94: 910003fd    	mov	x29, sp
100000b98: 94000003    	bl	0x100000ba4 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000b9c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000ba0: d65f03c0    	ret

0000000100000ba4 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000ba4: 92800000    	mov	x0, #-0x1               ; =-1
100000ba8: d65f03c0    	ret

0000000100000bac <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000bac: d10043ff    	sub	sp, sp, #0x10
100000bb0: f90007e0    	str	x0, [sp, #0x8]
100000bb4: f94007e8    	ldr	x8, [sp, #0x8]
100000bb8: f1004108    	subs	x8, x8, #0x10
100000bbc: 1a9f97e0    	cset	w0, hi
100000bc0: 910043ff    	add	sp, sp, #0x10
100000bc4: d65f03c0    	ret

0000000100000bc8 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__sp_aligned_storageILm8EEEJmSt11align_val_tEEEPvDpT0_>:
100000bc8: d10083ff    	sub	sp, sp, #0x20
100000bcc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bd0: 910043fd    	add	x29, sp, #0x10
100000bd4: f90007e0    	str	x0, [sp, #0x8]
100000bd8: f90003e1    	str	x1, [sp]
100000bdc: f94007e0    	ldr	x0, [sp, #0x8]
100000be0: f94003e1    	ldr	x1, [sp]
100000be4: 940002bc    	bl	0x1000016d4 <___stack_chk_guard+0x1000016d4>
100000be8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000bec: 910083ff    	add	sp, sp, #0x20
100000bf0: d65f03c0    	ret

0000000100000bf4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__sp_aligned_storageILm8EEEEEPvm>:
100000bf4: d10083ff    	sub	sp, sp, #0x20
100000bf8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bfc: 910043fd    	add	x29, sp, #0x10
100000c00: f90007e0    	str	x0, [sp, #0x8]
100000c04: f94007e0    	ldr	x0, [sp, #0x8]
100000c08: 940002b6    	bl	0x1000016e0 <___stack_chk_guard+0x1000016e0>
100000c0c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000c10: 910083ff    	add	sp, sp, #0x20
100000c14: d65f03c0    	ret

0000000100000c18 <__ZNSt3__112construct_atB8ne200100INS_31__unbounded_array_control_blockIA_iNS_9allocatorIS2_EEEEJRKS4_RmEPS5_EEPT_SB_DpOT0_>:
100000c18: d100c3ff    	sub	sp, sp, #0x30
100000c1c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c20: 910083fd    	add	x29, sp, #0x20
100000c24: f81f83a0    	stur	x0, [x29, #-0x8]
100000c28: f9000be1    	str	x1, [sp, #0x10]
100000c2c: f90007e2    	str	x2, [sp, #0x8]
100000c30: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c34: f90003e0    	str	x0, [sp]
100000c38: f9400be1    	ldr	x1, [sp, #0x10]
100000c3c: f94007e8    	ldr	x8, [sp, #0x8]
100000c40: f9400102    	ldr	x2, [x8]
100000c44: 94000005    	bl	0x100000c58 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEEC1B8ne200100ERKS3_m>
100000c48: f94003e0    	ldr	x0, [sp]
100000c4c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c50: 9100c3ff    	add	sp, sp, #0x30
100000c54: d65f03c0    	ret

0000000100000c58 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEEC1B8ne200100ERKS3_m>:
100000c58: d100c3ff    	sub	sp, sp, #0x30
100000c5c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c60: 910083fd    	add	x29, sp, #0x20
100000c64: f81f83a0    	stur	x0, [x29, #-0x8]
100000c68: f9000be1    	str	x1, [sp, #0x10]
100000c6c: f90007e2    	str	x2, [sp, #0x8]
100000c70: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c74: f90003e0    	str	x0, [sp]
100000c78: f9400be1    	ldr	x1, [sp, #0x10]
100000c7c: f94007e2    	ldr	x2, [sp, #0x8]
100000c80: 94000005    	bl	0x100000c94 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEEC2B8ne200100ERKS3_m>
100000c84: f94003e0    	ldr	x0, [sp]
100000c88: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c8c: 9100c3ff    	add	sp, sp, #0x30
100000c90: d65f03c0    	ret

0000000100000c94 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEEC2B8ne200100ERKS3_m>:
100000c94: d10103ff    	sub	sp, sp, #0x40
100000c98: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000c9c: 9100c3fd    	add	x29, sp, #0x30
100000ca0: f81f83a0    	stur	x0, [x29, #-0x8]
100000ca4: f81f03a1    	stur	x1, [x29, #-0x10]
100000ca8: f9000fe2    	str	x2, [sp, #0x18]
100000cac: f85f83a0    	ldur	x0, [x29, #-0x8]
100000cb0: f90003e0    	str	x0, [sp]
100000cb4: d2800001    	mov	x1, #0x0                ; =0
100000cb8: 9400001c    	bl	0x100000d28 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000cbc: f94003e8    	ldr	x8, [sp]
100000cc0: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000cc4: 9102e129    	add	x9, x9, #0xb8
100000cc8: 91004129    	add	x9, x9, #0x10
100000ccc: f9000109    	str	x9, [x8]
100000cd0: f9400fe9    	ldr	x9, [sp, #0x18]
100000cd4: f9000d09    	str	x9, [x8, #0x18]
100000cd8: 91008100    	add	x0, x8, #0x20
100000cdc: 9400005d    	bl	0x100000e50 <__ZNSt3__15beginB8ne200100IiLm1EEEPT_RAT0__S1_>
100000ce0: aa0003e1    	mov	x1, x0
100000ce4: f94003e0    	ldr	x0, [sp]
100000ce8: f9400c02    	ldr	x2, [x0, #0x18]
100000cec: 94000022    	bl	0x100000d74 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_>
100000cf0: 14000001    	b	0x100000cf4 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEEC2B8ne200100ERKS3_m+0x60>
100000cf4: f94003e0    	ldr	x0, [sp]
100000cf8: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000cfc: 910103ff    	add	sp, sp, #0x40
100000d00: d65f03c0    	ret
100000d04: aa0003e8    	mov	x8, x0
100000d08: f94003e0    	ldr	x0, [sp]
100000d0c: f9000be8    	str	x8, [sp, #0x10]
100000d10: aa0103e8    	mov	x8, x1
100000d14: b9000fe8    	str	w8, [sp, #0xc]
100000d18: 94000275    	bl	0x1000016ec <___stack_chk_guard+0x1000016ec>
100000d1c: 14000001    	b	0x100000d20 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEEC2B8ne200100ERKS3_m+0x8c>
100000d20: f9400be0    	ldr	x0, [sp, #0x10]
100000d24: 94000257    	bl	0x100001680 <___stack_chk_guard+0x100001680>

0000000100000d28 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000d28: d100c3ff    	sub	sp, sp, #0x30
100000d2c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000d30: 910083fd    	add	x29, sp, #0x20
100000d34: f81f83a0    	stur	x0, [x29, #-0x8]
100000d38: f9000be1    	str	x1, [sp, #0x10]
100000d3c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000d40: f90007e0    	str	x0, [sp, #0x8]
100000d44: f9400be1    	ldr	x1, [sp, #0x10]
100000d48: 9400008b    	bl	0x100000f74 <__ZNSt3__114__shared_countC2B8ne200100El>
100000d4c: f94007e0    	ldr	x0, [sp, #0x8]
100000d50: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000d54: f9403908    	ldr	x8, [x8, #0x70]
100000d58: 91004108    	add	x8, x8, #0x10
100000d5c: f9000008    	str	x8, [x0]
100000d60: f9400be8    	ldr	x8, [sp, #0x10]
100000d64: f9000808    	str	x8, [x0, #0x10]
100000d68: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000d6c: 9100c3ff    	add	sp, sp, #0x30
100000d70: d65f03c0    	ret

0000000100000d74 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_>:
100000d74: d10283ff    	sub	sp, sp, #0xa0
100000d78: a9097bfd    	stp	x29, x30, [sp, #0x90]
100000d7c: 910243fd    	add	x29, sp, #0x90
100000d80: f81f83a0    	stur	x0, [x29, #-0x8]
100000d84: d10043a8    	sub	x8, x29, #0x10
100000d88: f9000fe8    	str	x8, [sp, #0x18]
100000d8c: f81f03a1    	stur	x1, [x29, #-0x10]
100000d90: f81e83a2    	stur	x2, [x29, #-0x18]
100000d94: f85f83a1    	ldur	x1, [x29, #-0x8]
100000d98: d10067a0    	sub	x0, x29, #0x19
100000d9c: f9000be0    	str	x0, [sp, #0x10]
100000da0: 94000081    	bl	0x100000fa4 <__ZNSt3__19allocatorIiEC1B8ne200100IA_iEERKNS0_IT_EE>
100000da4: f9400bea    	ldr	x10, [sp, #0x10]
100000da8: f9400fe8    	ldr	x8, [sp, #0x18]
100000dac: f85f03ab    	ldur	x11, [x29, #-0x10]
100000db0: d100a3a9    	sub	x9, x29, #0x28
100000db4: f81d83ab    	stur	x11, [x29, #-0x28]
100000db8: 9100c3e0    	add	x0, sp, #0x30
100000dbc: f9001bea    	str	x10, [sp, #0x30]
100000dc0: f9001fe9    	str	x9, [sp, #0x38]
100000dc4: f90023e8    	str	x8, [sp, #0x40]
100000dc8: 910123e8    	add	x8, sp, #0x48
100000dcc: 94000083    	bl	0x100000fd8 <__ZNSt3__122__make_exception_guardB8ne200100IZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_EENS_28__exception_guard_exceptionsIS6_EES6_>
100000dd0: 14000001    	b	0x100000dd4 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0x60>
100000dd4: f85e83a8    	ldur	x8, [x29, #-0x18]
100000dd8: b4000288    	cbz	x8, 0x100000e28 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0xb4>
100000ddc: 14000001    	b	0x100000de0 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0x6c>
100000de0: f85f03a1    	ldur	x1, [x29, #-0x10]
100000de4: d10067a0    	sub	x0, x29, #0x19
100000de8: 9400008f    	bl	0x100001024 <__ZNSt3__141__allocator_construct_at_multidimensionalB8ne200100INS_9allocatorIiEEiEEvRT_PT0_>
100000dec: 14000001    	b	0x100000df0 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0x7c>
100000df0: 14000001    	b	0x100000df4 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0x80>
100000df4: f85e83a8    	ldur	x8, [x29, #-0x18]
100000df8: f1000508    	subs	x8, x8, #0x1
100000dfc: f81e83a8    	stur	x8, [x29, #-0x18]
100000e00: f85f03a8    	ldur	x8, [x29, #-0x10]
100000e04: 91001108    	add	x8, x8, #0x4
100000e08: f81f03a8    	stur	x8, [x29, #-0x10]
100000e0c: 17fffff2    	b	0x100000dd4 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0x60>
100000e10: f90017e0    	str	x0, [sp, #0x28]
100000e14: aa0103e8    	mov	x8, x1
100000e18: b90027e8    	str	w8, [sp, #0x24]
100000e1c: 910123e0    	add	x0, sp, #0x48
100000e20: 94000093    	bl	0x10000106c <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED1B8ne200100Ev>
100000e24: 14000009    	b	0x100000e48 <__ZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_+0xd4>
100000e28: 910123e0    	add	x0, sp, #0x48
100000e2c: f90007e0    	str	x0, [sp, #0x8]
100000e30: 94000088    	bl	0x100001050 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_E10__completeB8ne200100Ev>
100000e34: f94007e0    	ldr	x0, [sp, #0x8]
100000e38: 9400008d    	bl	0x10000106c <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED1B8ne200100Ev>
100000e3c: a9497bfd    	ldp	x29, x30, [sp, #0x90]
100000e40: 910283ff    	add	sp, sp, #0xa0
100000e44: d65f03c0    	ret
100000e48: f94017e0    	ldr	x0, [sp, #0x28]
100000e4c: 9400020d    	bl	0x100001680 <___stack_chk_guard+0x100001680>

0000000100000e50 <__ZNSt3__15beginB8ne200100IiLm1EEEPT_RAT0__S1_>:
100000e50: d10043ff    	sub	sp, sp, #0x10
100000e54: f90007e0    	str	x0, [sp, #0x8]
100000e58: f94007e0    	ldr	x0, [sp, #0x8]
100000e5c: 910043ff    	add	sp, sp, #0x10
100000e60: d65f03c0    	ret

0000000100000e64 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED1Ev>:
100000e64: d10083ff    	sub	sp, sp, #0x20
100000e68: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e6c: 910043fd    	add	x29, sp, #0x10
100000e70: f90007e0    	str	x0, [sp, #0x8]
100000e74: f94007e0    	ldr	x0, [sp, #0x8]
100000e78: f90003e0    	str	x0, [sp]
100000e7c: 9400011c    	bl	0x1000012ec <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED2Ev>
100000e80: f94003e0    	ldr	x0, [sp]
100000e84: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e88: 910083ff    	add	sp, sp, #0x20
100000e8c: d65f03c0    	ret

0000000100000e90 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED0Ev>:
100000e90: d10083ff    	sub	sp, sp, #0x20
100000e94: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e98: 910043fd    	add	x29, sp, #0x10
100000e9c: f90007e0    	str	x0, [sp, #0x8]
100000ea0: f94007e0    	ldr	x0, [sp, #0x8]
100000ea4: f90003e0    	str	x0, [sp]
100000ea8: 97ffffef    	bl	0x100000e64 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED1Ev>
100000eac: f94003e0    	ldr	x0, [sp]
100000eb0: 94000212    	bl	0x1000016f8 <___stack_chk_guard+0x1000016f8>
100000eb4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000eb8: 910083ff    	add	sp, sp, #0x20
100000ebc: d65f03c0    	ret

0000000100000ec0 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
100000ec0: d100c3ff    	sub	sp, sp, #0x30
100000ec4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000ec8: 910083fd    	add	x29, sp, #0x20
100000ecc: f81f83a0    	stur	x0, [x29, #-0x8]
100000ed0: f85f83a1    	ldur	x1, [x29, #-0x8]
100000ed4: f90007e1    	str	x1, [sp, #0x8]
100000ed8: d10027a0    	sub	x0, x29, #0x9
100000edc: f90003e0    	str	x0, [sp]
100000ee0: 94000031    	bl	0x100000fa4 <__ZNSt3__19allocatorIiEC1B8ne200100IA_iEERKNS0_IT_EE>
100000ee4: f94003e0    	ldr	x0, [sp]
100000ee8: f94007e8    	ldr	x8, [sp, #0x8]
100000eec: f9400d09    	ldr	x9, [x8, #0x18]
100000ef0: 91008101    	add	x1, x8, #0x20
100000ef4: 91008108    	add	x8, x8, #0x20
100000ef8: 8b090902    	add	x2, x8, x9, lsl #2
100000efc: 940000d0    	bl	0x10000123c <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_>
100000f00: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000f04: 9100c3ff    	add	sp, sp, #0x30
100000f08: d65f03c0    	ret

0000000100000f0c <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
100000f0c: d10103ff    	sub	sp, sp, #0x40
100000f10: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000f14: 9100c3fd    	add	x29, sp, #0x30
100000f18: f81f83a0    	stur	x0, [x29, #-0x8]
100000f1c: f85f83a1    	ldur	x1, [x29, #-0x8]
100000f20: f90003e1    	str	x1, [sp]
100000f24: d10027a0    	sub	x0, x29, #0x9
100000f28: f90007e0    	str	x0, [sp, #0x8]
100000f2c: 97fffeaa    	bl	0x1000009d4 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEEC1B8ne200100IA_iEERKNS0_IT_EE>
100000f30: f94003e8    	ldr	x8, [sp]
100000f34: f9400d00    	ldr	x0, [x8, #0x18]
100000f38: 97fffe18    	bl	0x100000798 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE11__bytes_forB8ne200100Em>
100000f3c: f94003e8    	ldr	x8, [sp]
100000f40: f9000fe0    	str	x0, [sp, #0x18]
100000f44: f9000be8    	str	x8, [sp, #0x10]
100000f48: f9400be0    	ldr	x0, [sp, #0x10]
100000f4c: 94000100    	bl	0x10000134c <__ZNSt3__114pointer_traitsIPNS_20__sp_aligned_storageILm8EEEE10pointer_toB8ne200100ERS2_>
100000f50: aa0003e1    	mov	x1, x0
100000f54: f94007e0    	ldr	x0, [sp, #0x8]
100000f58: f9400fe8    	ldr	x8, [sp, #0x18]
100000f5c: d2800109    	mov	x9, #0x8                ; =8
100000f60: 9ac90902    	udiv	x2, x8, x9
100000f64: 940000ed    	bl	0x100001318 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE10deallocateB8ne200100ERS4_PS3_m>
100000f68: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000f6c: 910103ff    	add	sp, sp, #0x40
100000f70: d65f03c0    	ret

0000000100000f74 <__ZNSt3__114__shared_countC2B8ne200100El>:
100000f74: d10043ff    	sub	sp, sp, #0x10
100000f78: f90007e0    	str	x0, [sp, #0x8]
100000f7c: f90003e1    	str	x1, [sp]
100000f80: f94007e0    	ldr	x0, [sp, #0x8]
100000f84: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000f88: f9404108    	ldr	x8, [x8, #0x80]
100000f8c: 91004108    	add	x8, x8, #0x10
100000f90: f9000008    	str	x8, [x0]
100000f94: f94003e8    	ldr	x8, [sp]
100000f98: f9000408    	str	x8, [x0, #0x8]
100000f9c: 910043ff    	add	sp, sp, #0x10
100000fa0: d65f03c0    	ret

0000000100000fa4 <__ZNSt3__19allocatorIiEC1B8ne200100IA_iEERKNS0_IT_EE>:
100000fa4: d100c3ff    	sub	sp, sp, #0x30
100000fa8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000fac: 910083fd    	add	x29, sp, #0x20
100000fb0: f81f83a0    	stur	x0, [x29, #-0x8]
100000fb4: f9000be1    	str	x1, [sp, #0x10]
100000fb8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000fbc: f90007e0    	str	x0, [sp, #0x8]
100000fc0: f9400be1    	ldr	x1, [sp, #0x10]
100000fc4: 94000035    	bl	0x100001098 <__ZNSt3__19allocatorIiEC2B8ne200100IA_iEERKNS0_IT_EE>
100000fc8: f94007e0    	ldr	x0, [sp, #0x8]
100000fcc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000fd0: 9100c3ff    	add	sp, sp, #0x30
100000fd4: d65f03c0    	ret

0000000100000fd8 <__ZNSt3__122__make_exception_guardB8ne200100IZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_EENS_28__exception_guard_exceptionsIS6_EES6_>:
100000fd8: d10143ff    	sub	sp, sp, #0x50
100000fdc: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000fe0: 910103fd    	add	x29, sp, #0x40
100000fe4: f90007e8    	str	x8, [sp, #0x8]
100000fe8: aa0003e8    	mov	x8, x0
100000fec: f94007e0    	ldr	x0, [sp, #0x8]
100000ff0: aa0003e9    	mov	x9, x0
100000ff4: f81f83a9    	stur	x9, [x29, #-0x8]
100000ff8: aa0803e9    	mov	x9, x8
100000ffc: f81f03a9    	stur	x9, [x29, #-0x10]
100001000: 3dc00100    	ldr	q0, [x8]
100001004: 910043e1    	add	x1, sp, #0x10
100001008: 3d8007e0    	str	q0, [sp, #0x10]
10000100c: f9400908    	ldr	x8, [x8, #0x10]
100001010: f90013e8    	str	x8, [sp, #0x20]
100001014: 94000032    	bl	0x1000010dc <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_EC1B8ne200100ESA_>
100001018: a9447bfd    	ldp	x29, x30, [sp, #0x40]
10000101c: 910143ff    	add	sp, sp, #0x50
100001020: d65f03c0    	ret

0000000100001024 <__ZNSt3__141__allocator_construct_at_multidimensionalB8ne200100INS_9allocatorIiEEiEEvRT_PT0_>:
100001024: d10083ff    	sub	sp, sp, #0x20
100001028: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000102c: 910043fd    	add	x29, sp, #0x10
100001030: f90007e0    	str	x0, [sp, #0x8]
100001034: f90003e1    	str	x1, [sp]
100001038: f94007e0    	ldr	x0, [sp, #0x8]
10000103c: f94003e1    	ldr	x1, [sp]
100001040: 940001b1    	bl	0x100001704 <___stack_chk_guard+0x100001704>
100001044: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001048: 910083ff    	add	sp, sp, #0x20
10000104c: d65f03c0    	ret

0000000100001050 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_E10__completeB8ne200100Ev>:
100001050: d10043ff    	sub	sp, sp, #0x10
100001054: f90007e0    	str	x0, [sp, #0x8]
100001058: f94007e9    	ldr	x9, [sp, #0x8]
10000105c: 52800028    	mov	w8, #0x1                ; =1
100001060: 39006128    	strb	w8, [x9, #0x18]
100001064: 910043ff    	add	sp, sp, #0x10
100001068: d65f03c0    	ret

000000010000106c <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED1B8ne200100Ev>:
10000106c: d10083ff    	sub	sp, sp, #0x20
100001070: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001074: 910043fd    	add	x29, sp, #0x10
100001078: f90007e0    	str	x0, [sp, #0x8]
10000107c: f94007e0    	ldr	x0, [sp, #0x8]
100001080: f90003e0    	str	x0, [sp]
100001084: 94000048    	bl	0x1000011a4 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED2B8ne200100Ev>
100001088: f94003e0    	ldr	x0, [sp]
10000108c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001090: 910083ff    	add	sp, sp, #0x20
100001094: d65f03c0    	ret

0000000100001098 <__ZNSt3__19allocatorIiEC2B8ne200100IA_iEERKNS0_IT_EE>:
100001098: d100c3ff    	sub	sp, sp, #0x30
10000109c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000010a0: 910083fd    	add	x29, sp, #0x20
1000010a4: f81f83a0    	stur	x0, [x29, #-0x8]
1000010a8: f9000be1    	str	x1, [sp, #0x10]
1000010ac: f85f83a0    	ldur	x0, [x29, #-0x8]
1000010b0: f90007e0    	str	x0, [sp, #0x8]
1000010b4: 94000005    	bl	0x1000010c8 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
1000010b8: f94007e0    	ldr	x0, [sp, #0x8]
1000010bc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000010c0: 9100c3ff    	add	sp, sp, #0x30
1000010c4: d65f03c0    	ret

00000001000010c8 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
1000010c8: d10043ff    	sub	sp, sp, #0x10
1000010cc: f90007e0    	str	x0, [sp, #0x8]
1000010d0: f94007e0    	ldr	x0, [sp, #0x8]
1000010d4: 910043ff    	add	sp, sp, #0x10
1000010d8: d65f03c0    	ret

00000001000010dc <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_EC1B8ne200100ESA_>:
1000010dc: d100c3ff    	sub	sp, sp, #0x30
1000010e0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000010e4: 910083fd    	add	x29, sp, #0x20
1000010e8: f81f83a0    	stur	x0, [x29, #-0x8]
1000010ec: aa0103e8    	mov	x8, x1
1000010f0: f9000be8    	str	x8, [sp, #0x10]
1000010f4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000010f8: f90007e0    	str	x0, [sp, #0x8]
1000010fc: 94000005    	bl	0x100001110 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_EC2B8ne200100ESA_>
100001100: f94007e0    	ldr	x0, [sp, #0x8]
100001104: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001108: 9100c3ff    	add	sp, sp, #0x30
10000110c: d65f03c0    	ret

0000000100001110 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_EC2B8ne200100ESA_>:
100001110: d10043ff    	sub	sp, sp, #0x10
100001114: f90007e0    	str	x0, [sp, #0x8]
100001118: aa0103e8    	mov	x8, x1
10000111c: f90003e8    	str	x8, [sp]
100001120: f94007e0    	ldr	x0, [sp, #0x8]
100001124: 3dc00020    	ldr	q0, [x1]
100001128: 3d800000    	str	q0, [x0]
10000112c: f9400828    	ldr	x8, [x1, #0x10]
100001130: f9000808    	str	x8, [x0, #0x10]
100001134: 3900601f    	strb	wzr, [x0, #0x18]
100001138: 910043ff    	add	sp, sp, #0x10
10000113c: d65f03c0    	ret

0000000100001140 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE9constructB8ne200100IiJEvLi0EEEvRS2_PT_DpOT0_>:
100001140: d10083ff    	sub	sp, sp, #0x20
100001144: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001148: 910043fd    	add	x29, sp, #0x10
10000114c: f90007e0    	str	x0, [sp, #0x8]
100001150: f90003e1    	str	x1, [sp]
100001154: f94003e0    	ldr	x0, [sp]
100001158: 94000004    	bl	0x100001168 <__ZNSt3__114__construct_atB8ne200100IiJEPiEEPT_S3_DpOT0_>
10000115c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001160: 910083ff    	add	sp, sp, #0x20
100001164: d65f03c0    	ret

0000000100001168 <__ZNSt3__114__construct_atB8ne200100IiJEPiEEPT_S3_DpOT0_>:
100001168: d10083ff    	sub	sp, sp, #0x20
10000116c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001170: 910043fd    	add	x29, sp, #0x10
100001174: f90007e0    	str	x0, [sp, #0x8]
100001178: f94007e0    	ldr	x0, [sp, #0x8]
10000117c: 94000004    	bl	0x10000118c <__ZNSt3__112construct_atB8ne200100IiJEPiEEPT_S3_DpOT0_>
100001180: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001184: 910083ff    	add	sp, sp, #0x20
100001188: d65f03c0    	ret

000000010000118c <__ZNSt3__112construct_atB8ne200100IiJEPiEEPT_S3_DpOT0_>:
10000118c: d10043ff    	sub	sp, sp, #0x10
100001190: f90007e0    	str	x0, [sp, #0x8]
100001194: f94007e0    	ldr	x0, [sp, #0x8]
100001198: b900001f    	str	wzr, [x0]
10000119c: 910043ff    	add	sp, sp, #0x10
1000011a0: d65f03c0    	ret

00000001000011a4 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED2B8ne200100Ev>:
1000011a4: d100c3ff    	sub	sp, sp, #0x30
1000011a8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000011ac: 910083fd    	add	x29, sp, #0x20
1000011b0: f9000be0    	str	x0, [sp, #0x10]
1000011b4: f9400be8    	ldr	x8, [sp, #0x10]
1000011b8: f90007e8    	str	x8, [sp, #0x8]
1000011bc: aa0803e9    	mov	x9, x8
1000011c0: f81f83a9    	stur	x9, [x29, #-0x8]
1000011c4: 39406108    	ldrb	w8, [x8, #0x18]
1000011c8: 370000c8    	tbnz	w8, #0x0, 0x1000011e0 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED2B8ne200100Ev+0x3c>
1000011cc: 14000001    	b	0x1000011d0 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED2B8ne200100Ev+0x2c>
1000011d0: f94007e0    	ldr	x0, [sp, #0x8]
1000011d4: 94000008    	bl	0x1000011f4 <__ZZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_ENKUlvE_clEv>
1000011d8: 14000001    	b	0x1000011dc <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED2B8ne200100Ev+0x38>
1000011dc: 14000001    	b	0x1000011e0 <__ZNSt3__128__exception_guard_exceptionsIZNS_60__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_EUlvE_ED2B8ne200100Ev+0x3c>
1000011e0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000011e4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000011e8: 9100c3ff    	add	sp, sp, #0x30
1000011ec: d65f03c0    	ret
1000011f0: 9400000f    	bl	0x10000122c <___clang_call_terminate>

00000001000011f4 <__ZZNSt3__160__uninitialized_allocator_value_construct_n_multidimensionalB8ne200100INS_9allocatorIA_iEEPimEEvRT_T0_T1_ENKUlvE_clEv>:
1000011f4: d10083ff    	sub	sp, sp, #0x20
1000011f8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000011fc: 910043fd    	add	x29, sp, #0x10
100001200: f90007e0    	str	x0, [sp, #0x8]
100001204: f94007e8    	ldr	x8, [sp, #0x8]
100001208: f9400100    	ldr	x0, [x8]
10000120c: f9400509    	ldr	x9, [x8, #0x8]
100001210: f9400121    	ldr	x1, [x9]
100001214: f9400908    	ldr	x8, [x8, #0x10]
100001218: f9400102    	ldr	x2, [x8]
10000121c: 94000008    	bl	0x10000123c <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_>
100001220: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001224: 910083ff    	add	sp, sp, #0x20
100001228: d65f03c0    	ret

000000010000122c <___clang_call_terminate>:
10000122c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100001230: 910003fd    	mov	x29, sp
100001234: 94000137    	bl	0x100001710 <___stack_chk_guard+0x100001710>
100001238: 94000139    	bl	0x10000171c <___stack_chk_guard+0x10000171c>

000000010000123c <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_>:
10000123c: d100c3ff    	sub	sp, sp, #0x30
100001240: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001244: 910083fd    	add	x29, sp, #0x20
100001248: f81f83a0    	stur	x0, [x29, #-0x8]
10000124c: f9000be1    	str	x1, [sp, #0x10]
100001250: f90007e2    	str	x2, [sp, #0x8]
100001254: f9400be8    	ldr	x8, [sp, #0x10]
100001258: f94007e9    	ldr	x9, [sp, #0x8]
10000125c: eb090108    	subs	x8, x8, x9
100001260: 54000061    	b.ne	0x10000126c <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x30>
100001264: 14000001    	b	0x100001268 <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x2c>
100001268: 1400000f    	b	0x1000012a4 <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x68>
10000126c: 14000001    	b	0x100001270 <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x34>
100001270: f94007e8    	ldr	x8, [sp, #0x8]
100001274: f1001108    	subs	x8, x8, #0x4
100001278: f90007e8    	str	x8, [sp, #0x8]
10000127c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001280: f94007e1    	ldr	x1, [sp, #0x8]
100001284: 94000129    	bl	0x100001728 <___stack_chk_guard+0x100001728>
100001288: 14000001    	b	0x10000128c <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x50>
10000128c: 14000001    	b	0x100001290 <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x54>
100001290: f94007e8    	ldr	x8, [sp, #0x8]
100001294: f9400be9    	ldr	x9, [sp, #0x10]
100001298: eb090108    	subs	x8, x8, x9
10000129c: 54fffea1    	b.ne	0x100001270 <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x34>
1000012a0: 14000001    	b	0x1000012a4 <__ZNSt3__136__allocator_destroy_multidimensionalB8ne200100INS_9allocatorIiEEPiLi0EEEvRT_T0_S6_+0x68>
1000012a4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000012a8: 9100c3ff    	add	sp, sp, #0x30
1000012ac: d65f03c0    	ret
1000012b0: 97ffffdf    	bl	0x10000122c <___clang_call_terminate>

00000001000012b4 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE7destroyB8ne200100IivLi0EEEvRS2_PT_>:
1000012b4: d10083ff    	sub	sp, sp, #0x20
1000012b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012bc: 910043fd    	add	x29, sp, #0x10
1000012c0: f90007e0    	str	x0, [sp, #0x8]
1000012c4: f90003e1    	str	x1, [sp]
1000012c8: f94003e0    	ldr	x0, [sp]
1000012cc: 94000004    	bl	0x1000012dc <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>
1000012d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012d4: 910083ff    	add	sp, sp, #0x20
1000012d8: d65f03c0    	ret

00000001000012dc <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>:
1000012dc: d10043ff    	sub	sp, sp, #0x10
1000012e0: f90007e0    	str	x0, [sp, #0x8]
1000012e4: 910043ff    	add	sp, sp, #0x10
1000012e8: d65f03c0    	ret

00000001000012ec <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED2Ev>:
1000012ec: d10083ff    	sub	sp, sp, #0x20
1000012f0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012f4: 910043fd    	add	x29, sp, #0x10
1000012f8: f90007e0    	str	x0, [sp, #0x8]
1000012fc: f94007e0    	ldr	x0, [sp, #0x8]
100001300: f90003e0    	str	x0, [sp]
100001304: 940000fa    	bl	0x1000016ec <___stack_chk_guard+0x1000016ec>
100001308: f94003e0    	ldr	x0, [sp]
10000130c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001310: 910083ff    	add	sp, sp, #0x20
100001314: d65f03c0    	ret

0000000100001318 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE10deallocateB8ne200100ERS4_PS3_m>:
100001318: d100c3ff    	sub	sp, sp, #0x30
10000131c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001320: 910083fd    	add	x29, sp, #0x20
100001324: f81f83a0    	stur	x0, [x29, #-0x8]
100001328: f9000be1    	str	x1, [sp, #0x10]
10000132c: f90007e2    	str	x2, [sp, #0x8]
100001330: f85f83a0    	ldur	x0, [x29, #-0x8]
100001334: f9400be1    	ldr	x1, [sp, #0x10]
100001338: f94007e2    	ldr	x2, [sp, #0x8]
10000133c: 94000009    	bl	0x100001360 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEE10deallocateB8ne200100EPS2_m>
100001340: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001344: 9100c3ff    	add	sp, sp, #0x30
100001348: d65f03c0    	ret

000000010000134c <__ZNSt3__114pointer_traitsIPNS_20__sp_aligned_storageILm8EEEE10pointer_toB8ne200100ERS2_>:
10000134c: d10043ff    	sub	sp, sp, #0x10
100001350: f90007e0    	str	x0, [sp, #0x8]
100001354: f94007e0    	ldr	x0, [sp, #0x8]
100001358: 910043ff    	add	sp, sp, #0x10
10000135c: d65f03c0    	ret

0000000100001360 <__ZNSt3__19allocatorINS_20__sp_aligned_storageILm8EEEE10deallocateB8ne200100EPS2_m>:
100001360: d100c3ff    	sub	sp, sp, #0x30
100001364: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001368: 910083fd    	add	x29, sp, #0x20
10000136c: f81f83a0    	stur	x0, [x29, #-0x8]
100001370: f9000be1    	str	x1, [sp, #0x10]
100001374: f90007e2    	str	x2, [sp, #0x8]
100001378: f9400be0    	ldr	x0, [sp, #0x10]
10000137c: f94007e1    	ldr	x1, [sp, #0x8]
100001380: d2800102    	mov	x2, #0x8                ; =8
100001384: 94000004    	bl	0x100001394 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
100001388: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000138c: 9100c3ff    	add	sp, sp, #0x30
100001390: d65f03c0    	ret

0000000100001394 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
100001394: d10103ff    	sub	sp, sp, #0x40
100001398: a9037bfd    	stp	x29, x30, [sp, #0x30]
10000139c: 9100c3fd    	add	x29, sp, #0x30
1000013a0: f81f83a0    	stur	x0, [x29, #-0x8]
1000013a4: f81f03a1    	stur	x1, [x29, #-0x10]
1000013a8: f9000fe2    	str	x2, [sp, #0x18]
1000013ac: f85f03a8    	ldur	x8, [x29, #-0x10]
1000013b0: d37df108    	lsl	x8, x8, #3
1000013b4: f9000be8    	str	x8, [sp, #0x10]
1000013b8: f9400fe0    	ldr	x0, [sp, #0x18]
1000013bc: 97fffdfc    	bl	0x100000bac <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
1000013c0: 36000100    	tbz	w0, #0x0, 0x1000013e0 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x4c>
1000013c4: 14000001    	b	0x1000013c8 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x34>
1000013c8: f9400fe8    	ldr	x8, [sp, #0x18]
1000013cc: f90007e8    	str	x8, [sp, #0x8]
1000013d0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000013d4: f94007e1    	ldr	x1, [sp, #0x8]
1000013d8: 94000008    	bl	0x1000013f8 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__sp_aligned_storageILm8EEESt11align_val_tEEEvDpT_>
1000013dc: 14000004    	b	0x1000013ec <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
1000013e0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000013e4: 94000010    	bl	0x100001424 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__sp_aligned_storageILm8EEEEEEvDpT_>
1000013e8: 14000001    	b	0x1000013ec <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__sp_aligned_storageILm8EEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
1000013ec: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000013f0: 910103ff    	add	sp, sp, #0x40
1000013f4: d65f03c0    	ret

00000001000013f8 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__sp_aligned_storageILm8EEESt11align_val_tEEEvDpT_>:
1000013f8: d10083ff    	sub	sp, sp, #0x20
1000013fc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001400: 910043fd    	add	x29, sp, #0x10
100001404: f90007e0    	str	x0, [sp, #0x8]
100001408: f90003e1    	str	x1, [sp]
10000140c: f94007e0    	ldr	x0, [sp, #0x8]
100001410: f94003e1    	ldr	x1, [sp]
100001414: 940000c8    	bl	0x100001734 <___stack_chk_guard+0x100001734>
100001418: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000141c: 910083ff    	add	sp, sp, #0x20
100001420: d65f03c0    	ret

0000000100001424 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__sp_aligned_storageILm8EEEEEEvDpT_>:
100001424: d10083ff    	sub	sp, sp, #0x20
100001428: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000142c: 910043fd    	add	x29, sp, #0x10
100001430: f90007e0    	str	x0, [sp, #0x8]
100001434: f94007e0    	ldr	x0, [sp, #0x8]
100001438: 940000b0    	bl	0x1000016f8 <___stack_chk_guard+0x1000016f8>
10000143c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001440: 910083ff    	add	sp, sp, #0x20
100001444: d65f03c0    	ret

0000000100001448 <__ZNSt3__110shared_ptrIA_iEC1B8ne200100Ev>:
100001448: d10083ff    	sub	sp, sp, #0x20
10000144c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001450: 910043fd    	add	x29, sp, #0x10
100001454: f90007e0    	str	x0, [sp, #0x8]
100001458: f94007e0    	ldr	x0, [sp, #0x8]
10000145c: f90003e0    	str	x0, [sp]
100001460: 94000009    	bl	0x100001484 <__ZNSt3__110shared_ptrIA_iEC2B8ne200100Ev>
100001464: f94003e0    	ldr	x0, [sp]
100001468: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000146c: 910083ff    	add	sp, sp, #0x20
100001470: d65f03c0    	ret

0000000100001474 <__ZNSt3__110shared_ptrIA_iE18__enable_weak_thisB8ne200100Ez>:
100001474: d10043ff    	sub	sp, sp, #0x10
100001478: f90007e0    	str	x0, [sp, #0x8]
10000147c: 910043ff    	add	sp, sp, #0x10
100001480: d65f03c0    	ret

0000000100001484 <__ZNSt3__110shared_ptrIA_iEC2B8ne200100Ev>:
100001484: d10043ff    	sub	sp, sp, #0x10
100001488: f90007e0    	str	x0, [sp, #0x8]
10000148c: f94007e0    	ldr	x0, [sp, #0x8]
100001490: f900001f    	str	xzr, [x0]
100001494: f900041f    	str	xzr, [x0, #0x8]
100001498: 910043ff    	add	sp, sp, #0x10
10000149c: d65f03c0    	ret

00000001000014a0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEED2B8ne200100Ev>:
1000014a0: d10083ff    	sub	sp, sp, #0x20
1000014a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000014a8: 910043fd    	add	x29, sp, #0x10
1000014ac: f90007e0    	str	x0, [sp, #0x8]
1000014b0: f94007e0    	ldr	x0, [sp, #0x8]
1000014b4: f90003e0    	str	x0, [sp]
1000014b8: 94000005    	bl	0x1000014cc <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE9__destroyB8ne200100Ev>
1000014bc: f94003e0    	ldr	x0, [sp]
1000014c0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000014c4: 910083ff    	add	sp, sp, #0x20
1000014c8: d65f03c0    	ret

00000001000014cc <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE9__destroyB8ne200100Ev>:
1000014cc: d10083ff    	sub	sp, sp, #0x20
1000014d0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000014d4: 910043fd    	add	x29, sp, #0x10
1000014d8: f90007e0    	str	x0, [sp, #0x8]
1000014dc: f94007e8    	ldr	x8, [sp, #0x8]
1000014e0: f90003e8    	str	x8, [sp]
1000014e4: f9400908    	ldr	x8, [x8, #0x10]
1000014e8: b40000e8    	cbz	x8, 0x100001504 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE9__destroyB8ne200100Ev+0x38>
1000014ec: 14000001    	b	0x1000014f0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE9__destroyB8ne200100Ev+0x24>
1000014f0: f94003e0    	ldr	x0, [sp]
1000014f4: f9400801    	ldr	x1, [x0, #0x10]
1000014f8: f9400402    	ldr	x2, [x0, #0x8]
1000014fc: 97ffff87    	bl	0x100001318 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE10deallocateB8ne200100ERS4_PS3_m>
100001500: 14000001    	b	0x100001504 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__sp_aligned_storageILm8EEEEEE9__destroyB8ne200100Ev+0x38>
100001504: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001508: 910083ff    	add	sp, sp, #0x20
10000150c: d65f03c0    	ret

0000000100001510 <__ZNSt3__19allocatorIA_iEC2B8ne200100Ev>:
100001510: d10083ff    	sub	sp, sp, #0x20
100001514: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001518: 910043fd    	add	x29, sp, #0x10
10000151c: f90007e0    	str	x0, [sp, #0x8]
100001520: f94007e0    	ldr	x0, [sp, #0x8]
100001524: f90003e0    	str	x0, [sp]
100001528: 94000005    	bl	0x10000153c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIA_iEEEC2B8ne200100Ev>
10000152c: f94003e0    	ldr	x0, [sp]
100001530: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001534: 910083ff    	add	sp, sp, #0x20
100001538: d65f03c0    	ret

000000010000153c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIA_iEEEC2B8ne200100Ev>:
10000153c: d10043ff    	sub	sp, sp, #0x10
100001540: f90007e0    	str	x0, [sp, #0x8]
100001544: f94007e0    	ldr	x0, [sp, #0x8]
100001548: 910043ff    	add	sp, sp, #0x10
10000154c: d65f03c0    	ret

0000000100001550 <__ZNSt3__110shared_ptrIA_iED2B8ne200100Ev>:
100001550: d100c3ff    	sub	sp, sp, #0x30
100001554: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001558: 910083fd    	add	x29, sp, #0x20
10000155c: f9000be0    	str	x0, [sp, #0x10]
100001560: f9400be8    	ldr	x8, [sp, #0x10]
100001564: f90007e8    	str	x8, [sp, #0x8]
100001568: aa0803e9    	mov	x9, x8
10000156c: f81f83a9    	stur	x9, [x29, #-0x8]
100001570: f9400508    	ldr	x8, [x8, #0x8]
100001574: b40000c8    	cbz	x8, 0x10000158c <__ZNSt3__110shared_ptrIA_iED2B8ne200100Ev+0x3c>
100001578: 14000001    	b	0x10000157c <__ZNSt3__110shared_ptrIA_iED2B8ne200100Ev+0x2c>
10000157c: f94007e8    	ldr	x8, [sp, #0x8]
100001580: f9400500    	ldr	x0, [x8, #0x8]
100001584: 94000006    	bl	0x10000159c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
100001588: 14000001    	b	0x10000158c <__ZNSt3__110shared_ptrIA_iED2B8ne200100Ev+0x3c>
10000158c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001590: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001594: 9100c3ff    	add	sp, sp, #0x30
100001598: d65f03c0    	ret

000000010000159c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
10000159c: d10083ff    	sub	sp, sp, #0x20
1000015a0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000015a4: 910043fd    	add	x29, sp, #0x10
1000015a8: f90007e0    	str	x0, [sp, #0x8]
1000015ac: f94007e0    	ldr	x0, [sp, #0x8]
1000015b0: f90003e0    	str	x0, [sp]
1000015b4: 94000009    	bl	0x1000015d8 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
1000015b8: 360000a0    	tbz	w0, #0x0, 0x1000015cc <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
1000015bc: 14000001    	b	0x1000015c0 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
1000015c0: f94003e0    	ldr	x0, [sp]
1000015c4: 9400005f    	bl	0x100001740 <___stack_chk_guard+0x100001740>
1000015c8: 14000001    	b	0x1000015cc <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
1000015cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000015d0: 910083ff    	add	sp, sp, #0x20
1000015d4: d65f03c0    	ret

00000001000015d8 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
1000015d8: d100c3ff    	sub	sp, sp, #0x30
1000015dc: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000015e0: 910083fd    	add	x29, sp, #0x20
1000015e4: f9000be0    	str	x0, [sp, #0x10]
1000015e8: f9400be8    	ldr	x8, [sp, #0x10]
1000015ec: f90007e8    	str	x8, [sp, #0x8]
1000015f0: 91002100    	add	x0, x8, #0x8
1000015f4: 94000017    	bl	0x100001650 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
1000015f8: b1000408    	adds	x8, x0, #0x1
1000015fc: 54000161    	b.ne	0x100001628 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
100001600: 14000001    	b	0x100001604 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
100001604: f94007e0    	ldr	x0, [sp, #0x8]
100001608: f9400008    	ldr	x8, [x0]
10000160c: f9400908    	ldr	x8, [x8, #0x10]
100001610: d63f0100    	blr	x8
100001614: 52800028    	mov	w8, #0x1                ; =1
100001618: 12000108    	and	w8, w8, #0x1
10000161c: 12000108    	and	w8, w8, #0x1
100001620: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001624: 14000006    	b	0x10000163c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100001628: 52800008    	mov	w8, #0x0                ; =0
10000162c: 12000108    	and	w8, w8, #0x1
100001630: 12000108    	and	w8, w8, #0x1
100001634: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001638: 14000001    	b	0x10000163c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
10000163c: 385ff3a8    	ldurb	w8, [x29, #-0x1]
100001640: 12000100    	and	w0, w8, #0x1
100001644: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001648: 9100c3ff    	add	sp, sp, #0x30
10000164c: d65f03c0    	ret

0000000100001650 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
100001650: d10083ff    	sub	sp, sp, #0x20
100001654: f9000fe0    	str	x0, [sp, #0x18]
100001658: f9400fe8    	ldr	x8, [sp, #0x18]
10000165c: 92800009    	mov	x9, #-0x1               ; =-1
100001660: f9000be9    	str	x9, [sp, #0x10]
100001664: f9400be9    	ldr	x9, [sp, #0x10]
100001668: f8e90108    	ldaddal	x9, x8, [x8]
10000166c: 8b090108    	add	x8, x8, x9
100001670: f90007e8    	str	x8, [sp, #0x8]
100001674: f94007e0    	ldr	x0, [sp, #0x8]
100001678: 910083ff    	add	sp, sp, #0x20
10000167c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001680 <__stubs>:
100001680: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001684: f9400610    	ldr	x16, [x16, #0x8]
100001688: d61f0200    	br	x16
10000168c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001690: f9400e10    	ldr	x16, [x16, #0x18]
100001694: d61f0200    	br	x16
100001698: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000169c: f9401210    	ldr	x16, [x16, #0x20]
1000016a0: d61f0200    	br	x16
1000016a4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016a8: f9401610    	ldr	x16, [x16, #0x28]
1000016ac: d61f0200    	br	x16
1000016b0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016b4: f9401a10    	ldr	x16, [x16, #0x30]
1000016b8: d61f0200    	br	x16
1000016bc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016c0: f9401e10    	ldr	x16, [x16, #0x38]
1000016c4: d61f0200    	br	x16
1000016c8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016cc: f9402a10    	ldr	x16, [x16, #0x50]
1000016d0: d61f0200    	br	x16
1000016d4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016d8: f9402e10    	ldr	x16, [x16, #0x58]
1000016dc: d61f0200    	br	x16
1000016e0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016e4: f9403210    	ldr	x16, [x16, #0x60]
1000016e8: d61f0200    	br	x16
1000016ec: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016f0: f9403610    	ldr	x16, [x16, #0x68]
1000016f4: d61f0200    	br	x16
1000016f8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016fc: f9403e10    	ldr	x16, [x16, #0x78]
100001700: d61f0200    	br	x16
100001704: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001708: f9404610    	ldr	x16, [x16, #0x88]
10000170c: d61f0200    	br	x16
100001710: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001714: f9404a10    	ldr	x16, [x16, #0x90]
100001718: d61f0200    	br	x16
10000171c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001720: f9404e10    	ldr	x16, [x16, #0x98]
100001724: d61f0200    	br	x16
100001728: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000172c: f9405210    	ldr	x16, [x16, #0xa0]
100001730: d61f0200    	br	x16
100001734: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001738: f9405610    	ldr	x16, [x16, #0xa8]
10000173c: d61f0200    	br	x16
100001740: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001744: f9405a10    	ldr	x16, [x16, #0xb0]
100001748: d61f0200    	br	x16
