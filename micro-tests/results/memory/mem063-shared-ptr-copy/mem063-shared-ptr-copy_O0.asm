
/Users/jim/work/cppfort/micro-tests/results/memory/mem063-shared-ptr-copy/mem063-shared-ptr-copy_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z20test_shared_ptr_copyv>:
100000538: d10143ff    	sub	sp, sp, #0x50
10000053c: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000540: 910103fd    	add	x29, sp, #0x40
100000544: d10053a0    	sub	x0, x29, #0x14
100000548: 52800548    	mov	w8, #0x2a               ; =42
10000054c: b81ec3a8    	stur	w8, [x29, #-0x14]
100000550: d10043a8    	sub	x8, x29, #0x10
100000554: f90007e8    	str	x8, [sp, #0x8]
100000558: 94000012    	bl	0x1000005a0 <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>
10000055c: f94007e1    	ldr	x1, [sp, #0x8]
100000560: 910063e0    	add	x0, sp, #0x18
100000564: f90003e0    	str	x0, [sp]
100000568: 9400001e    	bl	0x1000005e0 <__ZNSt3__110shared_ptrIiEC1B8ne200100ERKS1_>
10000056c: f94003e0    	ldr	x0, [sp]
100000570: 94000029    	bl	0x100000614 <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>
100000574: aa0003e8    	mov	x8, x0
100000578: f94003e0    	ldr	x0, [sp]
10000057c: b9400108    	ldr	w8, [x8]
100000580: b90017e8    	str	w8, [sp, #0x14]
100000584: 9400002a    	bl	0x10000062c <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000588: f94007e0    	ldr	x0, [sp, #0x8]
10000058c: 94000028    	bl	0x10000062c <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000590: b94017e0    	ldr	w0, [sp, #0x14]
100000594: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000598: 910143ff    	add	sp, sp, #0x50
10000059c: d65f03c0    	ret

00000001000005a0 <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>:
1000005a0: d10103ff    	sub	sp, sp, #0x40
1000005a4: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000005a8: 9100c3fd    	add	x29, sp, #0x30
1000005ac: f9000be8    	str	x8, [sp, #0x10]
1000005b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000005b4: f81f03a0    	stur	x0, [x29, #-0x10]
1000005b8: d10047a0    	sub	x0, x29, #0x11
1000005bc: f90007e0    	str	x0, [sp, #0x8]
1000005c0: 94000065    	bl	0x100000754 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>
1000005c4: f94007e0    	ldr	x0, [sp, #0x8]
1000005c8: f9400be8    	ldr	x8, [sp, #0x10]
1000005cc: f85f03a1    	ldur	x1, [x29, #-0x10]
1000005d0: 9400002a    	bl	0x100000678 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>
1000005d4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000005d8: 910103ff    	add	sp, sp, #0x40
1000005dc: d65f03c0    	ret

00000001000005e0 <__ZNSt3__110shared_ptrIiEC1B8ne200100ERKS1_>:
1000005e0: d100c3ff    	sub	sp, sp, #0x30
1000005e4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005e8: 910083fd    	add	x29, sp, #0x20
1000005ec: f81f83a0    	stur	x0, [x29, #-0x8]
1000005f0: f9000be1    	str	x1, [sp, #0x10]
1000005f4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000005f8: f90007e0    	str	x0, [sp, #0x8]
1000005fc: f9400be1    	ldr	x1, [sp, #0x10]
100000600: 94000363    	bl	0x10000138c <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_>
100000604: f94007e0    	ldr	x0, [sp, #0x8]
100000608: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000060c: 9100c3ff    	add	sp, sp, #0x30
100000610: d65f03c0    	ret

0000000100000614 <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>:
100000614: d10043ff    	sub	sp, sp, #0x10
100000618: f90007e0    	str	x0, [sp, #0x8]
10000061c: f94007e8    	ldr	x8, [sp, #0x8]
100000620: f9400100    	ldr	x0, [x8]
100000624: 910043ff    	add	sp, sp, #0x10
100000628: d65f03c0    	ret

000000010000062c <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>:
10000062c: d10083ff    	sub	sp, sp, #0x20
100000630: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000634: 910043fd    	add	x29, sp, #0x10
100000638: f90007e0    	str	x0, [sp, #0x8]
10000063c: f94007e0    	ldr	x0, [sp, #0x8]
100000640: f90003e0    	str	x0, [sp]
100000644: 94000306    	bl	0x10000125c <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>
100000648: f94003e0    	ldr	x0, [sp]
10000064c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000650: 910083ff    	add	sp, sp, #0x20
100000654: d65f03c0    	ret

0000000100000658 <_main>:
100000658: d10083ff    	sub	sp, sp, #0x20
10000065c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000660: 910043fd    	add	x29, sp, #0x10
100000664: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000668: 97ffffb4    	bl	0x100000538 <__Z20test_shared_ptr_copyv>
10000066c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000670: 910083ff    	add	sp, sp, #0x20
100000674: d65f03c0    	ret

0000000100000678 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>:
100000678: d10203ff    	sub	sp, sp, #0x80
10000067c: a9077bfd    	stp	x29, x30, [sp, #0x70]
100000680: 9101c3fd    	add	x29, sp, #0x70
100000684: f9000be8    	str	x8, [sp, #0x10]
100000688: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
10000068c: f9400529    	ldr	x9, [x9, #0x8]
100000690: f9400129    	ldr	x9, [x9]
100000694: f81f83a9    	stur	x9, [x29, #-0x8]
100000698: f81d83a8    	stur	x8, [x29, #-0x28]
10000069c: f81d03a0    	stur	x0, [x29, #-0x30]
1000006a0: f9001fe1    	str	x1, [sp, #0x38]
1000006a4: d10083a0    	sub	x0, x29, #0x20
1000006a8: d2800021    	mov	x1, #0x1                ; =1
1000006ac: 94000035    	bl	0x100000780 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>
1000006b0: 14000001    	b	0x1000006b4 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x3c>
1000006b4: d10083a0    	sub	x0, x29, #0x20
1000006b8: 9400003f    	bl	0x1000007b4 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>
1000006bc: f9401fe1    	ldr	x1, [sp, #0x38]
1000006c0: 94000043    	bl	0x1000007cc <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>
1000006c4: 14000001    	b	0x1000006c8 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x50>
1000006c8: d10083a0    	sub	x0, x29, #0x20
1000006cc: f90007e0    	str	x0, [sp, #0x8]
1000006d0: 9400004c    	bl	0x100000800 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>
1000006d4: f9000fe0    	str	x0, [sp, #0x18]
1000006d8: f9400fe0    	ldr	x0, [sp, #0x18]
1000006dc: 9400007b    	bl	0x1000008c8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
1000006e0: f9400be8    	ldr	x8, [sp, #0x10]
1000006e4: f9400fe1    	ldr	x1, [sp, #0x18]
1000006e8: 94000362    	bl	0x100001470 <___stack_chk_guard+0x100001470>
1000006ec: f94007e0    	ldr	x0, [sp, #0x8]
1000006f0: 94000080    	bl	0x1000008f0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
1000006f4: f85f83a9    	ldur	x9, [x29, #-0x8]
1000006f8: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000006fc: f9400508    	ldr	x8, [x8, #0x8]
100000700: f9400108    	ldr	x8, [x8]
100000704: eb090108    	subs	x8, x8, x9
100000708: 54000060    	b.eq	0x100000714 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x9c>
10000070c: 14000001    	b	0x100000710 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x98>
100000710: 9400035b    	bl	0x10000147c <___stack_chk_guard+0x10000147c>
100000714: a9477bfd    	ldp	x29, x30, [sp, #0x70]
100000718: 910203ff    	add	sp, sp, #0x80
10000071c: d65f03c0    	ret
100000720: f90017e0    	str	x0, [sp, #0x28]
100000724: aa0103e8    	mov	x8, x1
100000728: b90027e8    	str	w8, [sp, #0x24]
10000072c: d10083a0    	sub	x0, x29, #0x20
100000730: 94000070    	bl	0x1000008f0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
100000734: 14000001    	b	0x100000738 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xc0>
100000738: f94017e0    	ldr	x0, [sp, #0x28]
10000073c: f90003e0    	str	x0, [sp]
100000740: 14000003    	b	0x10000074c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
100000744: f90003e0    	str	x0, [sp]
100000748: 14000001    	b	0x10000074c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
10000074c: f94003e0    	ldr	x0, [sp]
100000750: 9400034e    	bl	0x100001488 <___stack_chk_guard+0x100001488>

0000000100000754 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>:
100000754: d10083ff    	sub	sp, sp, #0x20
100000758: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000075c: 910043fd    	add	x29, sp, #0x10
100000760: f90007e0    	str	x0, [sp, #0x8]
100000764: f94007e0    	ldr	x0, [sp, #0x8]
100000768: f90003e0    	str	x0, [sp]
10000076c: 940002ac    	bl	0x10000121c <__ZNSt3__19allocatorIiEC2B8ne200100Ev>
100000770: f94003e0    	ldr	x0, [sp]
100000774: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000778: 910083ff    	add	sp, sp, #0x20
10000077c: d65f03c0    	ret

0000000100000780 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>:
100000780: d100c3ff    	sub	sp, sp, #0x30
100000784: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000788: 910083fd    	add	x29, sp, #0x20
10000078c: f9000be0    	str	x0, [sp, #0x10]
100000790: f90007e1    	str	x1, [sp, #0x8]
100000794: f9400be0    	ldr	x0, [sp, #0x10]
100000798: f90003e0    	str	x0, [sp]
10000079c: f94007e1    	ldr	x1, [sp, #0x8]
1000007a0: 9400005f    	bl	0x10000091c <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>
1000007a4: f94003e0    	ldr	x0, [sp]
1000007a8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000007ac: 9100c3ff    	add	sp, sp, #0x30
1000007b0: d65f03c0    	ret

00000001000007b4 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>:
1000007b4: d10043ff    	sub	sp, sp, #0x10
1000007b8: f90007e0    	str	x0, [sp, #0x8]
1000007bc: f94007e8    	ldr	x8, [sp, #0x8]
1000007c0: f9400900    	ldr	x0, [x8, #0x10]
1000007c4: 910043ff    	add	sp, sp, #0x10
1000007c8: d65f03c0    	ret

00000001000007cc <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>:
1000007cc: d100c3ff    	sub	sp, sp, #0x30
1000007d0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000007d4: 910083fd    	add	x29, sp, #0x20
1000007d8: f9000be0    	str	x0, [sp, #0x10]
1000007dc: f90007e1    	str	x1, [sp, #0x8]
1000007e0: f9400be0    	ldr	x0, [sp, #0x10]
1000007e4: f90003e0    	str	x0, [sp]
1000007e8: f94007e1    	ldr	x1, [sp, #0x8]
1000007ec: 940000f1    	bl	0x100000bb0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>
1000007f0: f94003e0    	ldr	x0, [sp]
1000007f4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000007f8: 9100c3ff    	add	sp, sp, #0x30
1000007fc: d65f03c0    	ret

0000000100000800 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>:
100000800: d10043ff    	sub	sp, sp, #0x10
100000804: f90007e0    	str	x0, [sp, #0x8]
100000808: f94007e8    	ldr	x8, [sp, #0x8]
10000080c: f9400909    	ldr	x9, [x8, #0x10]
100000810: f90003e9    	str	x9, [sp]
100000814: f900091f    	str	xzr, [x8, #0x10]
100000818: f94003e0    	ldr	x0, [sp]
10000081c: 910043ff    	add	sp, sp, #0x10
100000820: d65f03c0    	ret

0000000100000824 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_>:
100000824: d10143ff    	sub	sp, sp, #0x50
100000828: a9047bfd    	stp	x29, x30, [sp, #0x40]
10000082c: 910103fd    	add	x29, sp, #0x40
100000830: f9000fe8    	str	x8, [sp, #0x18]
100000834: aa0003e8    	mov	x8, x0
100000838: f9400fe0    	ldr	x0, [sp, #0x18]
10000083c: aa0003e9    	mov	x9, x0
100000840: f81f83a9    	stur	x9, [x29, #-0x8]
100000844: f81f03a8    	stur	x8, [x29, #-0x10]
100000848: f81e83a1    	stur	x1, [x29, #-0x18]
10000084c: 52800008    	mov	w8, #0x0                ; =0
100000850: 52800029    	mov	w9, #0x1                ; =1
100000854: b90023e9    	str	w9, [sp, #0x20]
100000858: 12000108    	and	w8, w8, #0x1
10000085c: 12000108    	and	w8, w8, #0x1
100000860: 381e73a8    	sturb	w8, [x29, #-0x19]
100000864: 94000237    	bl	0x100001140 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>
100000868: f9400fe0    	ldr	x0, [sp, #0x18]
10000086c: f85f03a8    	ldur	x8, [x29, #-0x10]
100000870: f9000008    	str	x8, [x0]
100000874: f85e83a8    	ldur	x8, [x29, #-0x18]
100000878: f9000408    	str	x8, [x0, #0x8]
10000087c: f940000a    	ldr	x10, [x0]
100000880: f9400008    	ldr	x8, [x0]
100000884: 910003e9    	mov	x9, sp
100000888: f900012a    	str	x10, [x9]
10000088c: f9000528    	str	x8, [x9, #0x8]
100000890: 94000237    	bl	0x10000116c <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>
100000894: b94023e9    	ldr	w9, [sp, #0x20]
100000898: 12000128    	and	w8, w9, #0x1
10000089c: 0a090108    	and	w8, w8, w9
1000008a0: 381e73a8    	sturb	w8, [x29, #-0x19]
1000008a4: 385e73a8    	ldurb	w8, [x29, #-0x19]
1000008a8: 370000a8    	tbnz	w8, #0x0, 0x1000008bc <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
1000008ac: 14000001    	b	0x1000008b0 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x8c>
1000008b0: f9400fe0    	ldr	x0, [sp, #0x18]
1000008b4: 97ffff5e    	bl	0x10000062c <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
1000008b8: 14000001    	b	0x1000008bc <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
1000008bc: a9447bfd    	ldp	x29, x30, [sp, #0x40]
1000008c0: 910143ff    	add	sp, sp, #0x50
1000008c4: d65f03c0    	ret

00000001000008c8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>:
1000008c8: d10083ff    	sub	sp, sp, #0x20
1000008cc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000008d0: 910043fd    	add	x29, sp, #0x10
1000008d4: f90007e0    	str	x0, [sp, #0x8]
1000008d8: f94007e8    	ldr	x8, [sp, #0x8]
1000008dc: 91006100    	add	x0, x8, #0x18
1000008e0: 9400022e    	bl	0x100001198 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>
1000008e4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000008e8: 910083ff    	add	sp, sp, #0x20
1000008ec: d65f03c0    	ret

00000001000008f0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>:
1000008f0: d10083ff    	sub	sp, sp, #0x20
1000008f4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000008f8: 910043fd    	add	x29, sp, #0x10
1000008fc: f90007e0    	str	x0, [sp, #0x8]
100000900: f94007e0    	ldr	x0, [sp, #0x8]
100000904: f90003e0    	str	x0, [sp]
100000908: 94000229    	bl	0x1000011ac <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>
10000090c: f94003e0    	ldr	x0, [sp]
100000910: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000914: 910083ff    	add	sp, sp, #0x20
100000918: d65f03c0    	ret

000000010000091c <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>:
10000091c: d100c3ff    	sub	sp, sp, #0x30
100000920: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000924: 910083fd    	add	x29, sp, #0x20
100000928: f9000be0    	str	x0, [sp, #0x10]
10000092c: f90007e1    	str	x1, [sp, #0x8]
100000930: f9400be0    	ldr	x0, [sp, #0x10]
100000934: f90003e0    	str	x0, [sp]
100000938: d10007a1    	sub	x1, x29, #0x1
10000093c: 9400000c    	bl	0x10000096c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000940: f94003e0    	ldr	x0, [sp]
100000944: f94007e8    	ldr	x8, [sp, #0x8]
100000948: f9000408    	str	x8, [x0, #0x8]
10000094c: f9400401    	ldr	x1, [x0, #0x8]
100000950: 94000014    	bl	0x1000009a0 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>
100000954: aa0003e8    	mov	x8, x0
100000958: f94003e0    	ldr	x0, [sp]
10000095c: f9000808    	str	x8, [x0, #0x10]
100000960: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000964: 9100c3ff    	add	sp, sp, #0x30
100000968: d65f03c0    	ret

000000010000096c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>:
10000096c: d100c3ff    	sub	sp, sp, #0x30
100000970: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000974: 910083fd    	add	x29, sp, #0x20
100000978: f81f83a0    	stur	x0, [x29, #-0x8]
10000097c: f9000be1    	str	x1, [sp, #0x10]
100000980: f85f83a0    	ldur	x0, [x29, #-0x8]
100000984: f90007e0    	str	x0, [sp, #0x8]
100000988: f9400be1    	ldr	x1, [sp, #0x10]
10000098c: 94000010    	bl	0x1000009cc <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>
100000990: f94007e0    	ldr	x0, [sp, #0x8]
100000994: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000998: 9100c3ff    	add	sp, sp, #0x30
10000099c: d65f03c0    	ret

00000001000009a0 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>:
1000009a0: d10083ff    	sub	sp, sp, #0x20
1000009a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000009a8: 910043fd    	add	x29, sp, #0x10
1000009ac: f90007e0    	str	x0, [sp, #0x8]
1000009b0: f90003e1    	str	x1, [sp]
1000009b4: f94007e0    	ldr	x0, [sp, #0x8]
1000009b8: f94003e1    	ldr	x1, [sp]
1000009bc: 94000015    	bl	0x100000a10 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>
1000009c0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000009c4: 910083ff    	add	sp, sp, #0x20
1000009c8: d65f03c0    	ret

00000001000009cc <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>:
1000009cc: d100c3ff    	sub	sp, sp, #0x30
1000009d0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000009d4: 910083fd    	add	x29, sp, #0x20
1000009d8: f81f83a0    	stur	x0, [x29, #-0x8]
1000009dc: f9000be1    	str	x1, [sp, #0x10]
1000009e0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000009e4: f90007e0    	str	x0, [sp, #0x8]
1000009e8: 94000005    	bl	0x1000009fc <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>
1000009ec: f94007e0    	ldr	x0, [sp, #0x8]
1000009f0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000009f4: 9100c3ff    	add	sp, sp, #0x30
1000009f8: d65f03c0    	ret

00000001000009fc <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>:
1000009fc: d10043ff    	sub	sp, sp, #0x10
100000a00: f90007e0    	str	x0, [sp, #0x8]
100000a04: f94007e0    	ldr	x0, [sp, #0x8]
100000a08: 910043ff    	add	sp, sp, #0x10
100000a0c: d65f03c0    	ret

0000000100000a10 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>:
100000a10: d100c3ff    	sub	sp, sp, #0x30
100000a14: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a18: 910083fd    	add	x29, sp, #0x20
100000a1c: f81f83a0    	stur	x0, [x29, #-0x8]
100000a20: f9000be1    	str	x1, [sp, #0x10]
100000a24: f85f83a0    	ldur	x0, [x29, #-0x8]
100000a28: f9400be8    	ldr	x8, [sp, #0x10]
100000a2c: f90007e8    	str	x8, [sp, #0x8]
100000a30: 94000299    	bl	0x100001494 <___stack_chk_guard+0x100001494>
100000a34: f94007e8    	ldr	x8, [sp, #0x8]
100000a38: eb000108    	subs	x8, x8, x0
100000a3c: 54000069    	b.ls	0x100000a48 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x38>
100000a40: 14000001    	b	0x100000a44 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x34>
100000a44: 94000011    	bl	0x100000a88 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
100000a48: f9400be0    	ldr	x0, [sp, #0x10]
100000a4c: d2800101    	mov	x1, #0x8                ; =8
100000a50: 9400001b    	bl	0x100000abc <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>
100000a54: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a58: 9100c3ff    	add	sp, sp, #0x30
100000a5c: d65f03c0    	ret

0000000100000a60 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8max_sizeB8ne200100IS5_vLi0EEEmRKS5_>:
100000a60: d10083ff    	sub	sp, sp, #0x20
100000a64: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a68: 910043fd    	add	x29, sp, #0x10
100000a6c: f90007e0    	str	x0, [sp, #0x8]
100000a70: 9400002e    	bl	0x100000b28 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000a74: d2800408    	mov	x8, #0x20               ; =32
100000a78: 9ac80800    	udiv	x0, x0, x8
100000a7c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a80: 910083ff    	add	sp, sp, #0x20
100000a84: d65f03c0    	ret

0000000100000a88 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000a88: d10083ff    	sub	sp, sp, #0x20
100000a8c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a90: 910043fd    	add	x29, sp, #0x10
100000a94: d2800100    	mov	x0, #0x8                ; =8
100000a98: 94000282    	bl	0x1000014a0 <___stack_chk_guard+0x1000014a0>
100000a9c: f90007e0    	str	x0, [sp, #0x8]
100000aa0: 94000283    	bl	0x1000014ac <___stack_chk_guard+0x1000014ac>
100000aa4: f94007e0    	ldr	x0, [sp, #0x8]
100000aa8: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000aac: f9402021    	ldr	x1, [x1, #0x40]
100000ab0: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000ab4: f9402442    	ldr	x2, [x2, #0x48]
100000ab8: 94000280    	bl	0x1000014b8 <___stack_chk_guard+0x1000014b8>

0000000100000abc <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>:
100000abc: d10103ff    	sub	sp, sp, #0x40
100000ac0: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000ac4: 9100c3fd    	add	x29, sp, #0x30
100000ac8: f81f03a0    	stur	x0, [x29, #-0x10]
100000acc: f9000fe1    	str	x1, [sp, #0x18]
100000ad0: f85f03a8    	ldur	x8, [x29, #-0x10]
100000ad4: d37be908    	lsl	x8, x8, #5
100000ad8: f9000be8    	str	x8, [sp, #0x10]
100000adc: f9400fe0    	ldr	x0, [sp, #0x18]
100000ae0: 94000019    	bl	0x100000b44 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000ae4: 36000120    	tbz	w0, #0x0, 0x100000b08 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x4c>
100000ae8: 14000001    	b	0x100000aec <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x30>
100000aec: f9400fe8    	ldr	x8, [sp, #0x18]
100000af0: f90007e8    	str	x8, [sp, #0x8]
100000af4: f9400be0    	ldr	x0, [sp, #0x10]
100000af8: f94007e1    	ldr	x1, [sp, #0x8]
100000afc: 94000019    	bl	0x100000b60 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>
100000b00: f81f83a0    	stur	x0, [x29, #-0x8]
100000b04: 14000005    	b	0x100000b18 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000b08: f9400be0    	ldr	x0, [sp, #0x10]
100000b0c: 94000020    	bl	0x100000b8c <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>
100000b10: f81f83a0    	stur	x0, [x29, #-0x8]
100000b14: 14000001    	b	0x100000b18 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000b18: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b1c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000b20: 910103ff    	add	sp, sp, #0x40
100000b24: d65f03c0    	ret

0000000100000b28 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000b28: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000b2c: 910003fd    	mov	x29, sp
100000b30: 94000003    	bl	0x100000b3c <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000b34: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000b38: d65f03c0    	ret

0000000100000b3c <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000b3c: 92800000    	mov	x0, #-0x1               ; =-1
100000b40: d65f03c0    	ret

0000000100000b44 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000b44: d10043ff    	sub	sp, sp, #0x10
100000b48: f90007e0    	str	x0, [sp, #0x8]
100000b4c: f94007e8    	ldr	x8, [sp, #0x8]
100000b50: f1004108    	subs	x8, x8, #0x10
100000b54: 1a9f97e0    	cset	w0, hi
100000b58: 910043ff    	add	sp, sp, #0x10
100000b5c: d65f03c0    	ret

0000000100000b60 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>:
100000b60: d10083ff    	sub	sp, sp, #0x20
100000b64: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000b68: 910043fd    	add	x29, sp, #0x10
100000b6c: f90007e0    	str	x0, [sp, #0x8]
100000b70: f90003e1    	str	x1, [sp]
100000b74: f94007e0    	ldr	x0, [sp, #0x8]
100000b78: f94003e1    	ldr	x1, [sp]
100000b7c: 94000252    	bl	0x1000014c4 <___stack_chk_guard+0x1000014c4>
100000b80: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000b84: 910083ff    	add	sp, sp, #0x20
100000b88: d65f03c0    	ret

0000000100000b8c <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>:
100000b8c: d10083ff    	sub	sp, sp, #0x20
100000b90: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000b94: 910043fd    	add	x29, sp, #0x10
100000b98: f90007e0    	str	x0, [sp, #0x8]
100000b9c: f94007e0    	ldr	x0, [sp, #0x8]
100000ba0: 9400024c    	bl	0x1000014d0 <___stack_chk_guard+0x1000014d0>
100000ba4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ba8: 910083ff    	add	sp, sp, #0x20
100000bac: d65f03c0    	ret

0000000100000bb0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000bb0: d10103ff    	sub	sp, sp, #0x40
100000bb4: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000bb8: 9100c3fd    	add	x29, sp, #0x30
100000bbc: f81f03a0    	stur	x0, [x29, #-0x10]
100000bc0: f9000fe1    	str	x1, [sp, #0x18]
100000bc4: f85f03a0    	ldur	x0, [x29, #-0x10]
100000bc8: f90003e0    	str	x0, [sp]
100000bcc: d2800001    	mov	x1, #0x0                ; =0
100000bd0: 94000027    	bl	0x100000c6c <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000bd4: f94003e8    	ldr	x8, [sp]
100000bd8: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000bdc: 91030129    	add	x9, x9, #0xc0
100000be0: 91004129    	add	x9, x9, #0x10
100000be4: f9000109    	str	x9, [x8]
100000be8: 91006100    	add	x0, x8, #0x18
100000bec: d10007a1    	sub	x1, x29, #0x1
100000bf0: 94000032    	bl	0x100000cb8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>
100000bf4: 14000001    	b	0x100000bf8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x48>
100000bf8: f94003e0    	ldr	x0, [sp]
100000bfc: 9400003c    	bl	0x100000cec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000c00: f94003e0    	ldr	x0, [sp]
100000c04: 97ffff31    	bl	0x1000008c8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000c08: aa0003e1    	mov	x1, x0
100000c0c: f9400fe2    	ldr	x2, [sp, #0x18]
100000c10: 91002fe0    	add	x0, sp, #0xb
100000c14: 94000232    	bl	0x1000014dc <___stack_chk_guard+0x1000014dc>
100000c18: 14000001    	b	0x100000c1c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x6c>
100000c1c: f94003e0    	ldr	x0, [sp]
100000c20: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000c24: 910103ff    	add	sp, sp, #0x40
100000c28: d65f03c0    	ret
100000c2c: f9000be0    	str	x0, [sp, #0x10]
100000c30: aa0103e8    	mov	x8, x1
100000c34: b9000fe8    	str	w8, [sp, #0xc]
100000c38: 14000008    	b	0x100000c58 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000c3c: f94003e8    	ldr	x8, [sp]
100000c40: f9000be0    	str	x0, [sp, #0x10]
100000c44: aa0103e9    	mov	x9, x1
100000c48: b9000fe9    	str	w9, [sp, #0xc]
100000c4c: 91006100    	add	x0, x8, #0x18
100000c50: 9400003d    	bl	0x100000d44 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000c54: 14000001    	b	0x100000c58 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000c58: f94003e0    	ldr	x0, [sp]
100000c5c: 94000223    	bl	0x1000014e8 <___stack_chk_guard+0x1000014e8>
100000c60: 14000001    	b	0x100000c64 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xb4>
100000c64: f9400be0    	ldr	x0, [sp, #0x10]
100000c68: 94000208    	bl	0x100001488 <___stack_chk_guard+0x100001488>

0000000100000c6c <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000c6c: d100c3ff    	sub	sp, sp, #0x30
100000c70: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c74: 910083fd    	add	x29, sp, #0x20
100000c78: f81f83a0    	stur	x0, [x29, #-0x8]
100000c7c: f9000be1    	str	x1, [sp, #0x10]
100000c80: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c84: f90007e0    	str	x0, [sp, #0x8]
100000c88: f9400be1    	ldr	x1, [sp, #0x10]
100000c8c: 94000070    	bl	0x100000e4c <__ZNSt3__114__shared_countC2B8ne200100El>
100000c90: f94007e0    	ldr	x0, [sp, #0x8]
100000c94: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000c98: f9403d08    	ldr	x8, [x8, #0x78]
100000c9c: 91004108    	add	x8, x8, #0x10
100000ca0: f9000008    	str	x8, [x0]
100000ca4: f9400be8    	ldr	x8, [sp, #0x10]
100000ca8: f9000808    	str	x8, [x0, #0x10]
100000cac: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000cb0: 9100c3ff    	add	sp, sp, #0x30
100000cb4: d65f03c0    	ret

0000000100000cb8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>:
100000cb8: d100c3ff    	sub	sp, sp, #0x30
100000cbc: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000cc0: 910083fd    	add	x29, sp, #0x20
100000cc4: f81f83a0    	stur	x0, [x29, #-0x8]
100000cc8: f9000be1    	str	x1, [sp, #0x10]
100000ccc: f85f83a0    	ldur	x0, [x29, #-0x8]
100000cd0: f90007e0    	str	x0, [sp, #0x8]
100000cd4: f9400be1    	ldr	x1, [sp, #0x10]
100000cd8: 94000069    	bl	0x100000e7c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>
100000cdc: f94007e0    	ldr	x0, [sp, #0x8]
100000ce0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ce4: 9100c3ff    	add	sp, sp, #0x30
100000ce8: d65f03c0    	ret

0000000100000cec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>:
100000cec: d10083ff    	sub	sp, sp, #0x20
100000cf0: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000cf4: 910043fd    	add	x29, sp, #0x10
100000cf8: f90007e0    	str	x0, [sp, #0x8]
100000cfc: f94007e8    	ldr	x8, [sp, #0x8]
100000d00: 91006100    	add	x0, x8, #0x18
100000d04: 9400006a    	bl	0x100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000d08: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d0c: 910083ff    	add	sp, sp, #0x20
100000d10: d65f03c0    	ret

0000000100000d14 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE9constructB8ne200100IiJiEvLi0EEEvRS2_PT_DpOT0_>:
100000d14: d100c3ff    	sub	sp, sp, #0x30
100000d18: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000d1c: 910083fd    	add	x29, sp, #0x20
100000d20: f81f83a0    	stur	x0, [x29, #-0x8]
100000d24: f9000be1    	str	x1, [sp, #0x10]
100000d28: f90007e2    	str	x2, [sp, #0x8]
100000d2c: f9400be0    	ldr	x0, [sp, #0x10]
100000d30: f94007e1    	ldr	x1, [sp, #0x8]
100000d34: 94000063    	bl	0x100000ec0 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000d38: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000d3c: 9100c3ff    	add	sp, sp, #0x30
100000d40: d65f03c0    	ret

0000000100000d44 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>:
100000d44: d10083ff    	sub	sp, sp, #0x20
100000d48: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d4c: 910043fd    	add	x29, sp, #0x10
100000d50: f90007e0    	str	x0, [sp, #0x8]
100000d54: f94007e0    	ldr	x0, [sp, #0x8]
100000d58: f90003e0    	str	x0, [sp]
100000d5c: 9400006d    	bl	0x100000f10 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>
100000d60: f94003e0    	ldr	x0, [sp]
100000d64: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d68: 910083ff    	add	sp, sp, #0x20
100000d6c: d65f03c0    	ret

0000000100000d70 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000d70: d10083ff    	sub	sp, sp, #0x20
100000d74: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d78: 910043fd    	add	x29, sp, #0x10
100000d7c: f90007e0    	str	x0, [sp, #0x8]
100000d80: f94007e0    	ldr	x0, [sp, #0x8]
100000d84: f90003e0    	str	x0, [sp]
100000d88: 9400006d    	bl	0x100000f3c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>
100000d8c: f94003e0    	ldr	x0, [sp]
100000d90: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d94: 910083ff    	add	sp, sp, #0x20
100000d98: d65f03c0    	ret

0000000100000d9c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000d9c: d10083ff    	sub	sp, sp, #0x20
100000da0: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000da4: 910043fd    	add	x29, sp, #0x10
100000da8: f90007e0    	str	x0, [sp, #0x8]
100000dac: f94007e0    	ldr	x0, [sp, #0x8]
100000db0: f90003e0    	str	x0, [sp]
100000db4: 97ffffef    	bl	0x100000d70 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>
100000db8: f94003e0    	ldr	x0, [sp]
100000dbc: 940001ce    	bl	0x1000014f4 <___stack_chk_guard+0x1000014f4>
100000dc0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000dc4: 910083ff    	add	sp, sp, #0x20
100000dc8: d65f03c0    	ret

0000000100000dcc <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000dcc: d10083ff    	sub	sp, sp, #0x20
100000dd0: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000dd4: 910043fd    	add	x29, sp, #0x10
100000dd8: f90007e0    	str	x0, [sp, #0x8]
100000ddc: f94007e0    	ldr	x0, [sp, #0x8]
100000de0: 940001c8    	bl	0x100001500 <___stack_chk_guard+0x100001500>
100000de4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000de8: 910083ff    	add	sp, sp, #0x20
100000dec: d65f03c0    	ret

0000000100000df0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000df0: d100c3ff    	sub	sp, sp, #0x30
100000df4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000df8: 910083fd    	add	x29, sp, #0x20
100000dfc: f81f83a0    	stur	x0, [x29, #-0x8]
100000e00: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e04: f90003e0    	str	x0, [sp]
100000e08: 97ffffb9    	bl	0x100000cec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000e0c: aa0003e1    	mov	x1, x0
100000e10: d10027a0    	sub	x0, x29, #0x9
100000e14: f90007e0    	str	x0, [sp, #0x8]
100000e18: 97fffed5    	bl	0x10000096c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000e1c: f94003e8    	ldr	x8, [sp]
100000e20: 91006100    	add	x0, x8, #0x18
100000e24: 97ffffc8    	bl	0x100000d44 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000e28: f94003e0    	ldr	x0, [sp]
100000e2c: 94000086    	bl	0x100001044 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>
100000e30: aa0003e1    	mov	x1, x0
100000e34: f94007e0    	ldr	x0, [sp, #0x8]
100000e38: d2800022    	mov	x2, #0x1                ; =1
100000e3c: 94000075    	bl	0x100001010 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100000e40: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e44: 9100c3ff    	add	sp, sp, #0x30
100000e48: d65f03c0    	ret

0000000100000e4c <__ZNSt3__114__shared_countC2B8ne200100El>:
100000e4c: d10043ff    	sub	sp, sp, #0x10
100000e50: f90007e0    	str	x0, [sp, #0x8]
100000e54: f90003e1    	str	x1, [sp]
100000e58: f94007e0    	ldr	x0, [sp, #0x8]
100000e5c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000e60: f9404908    	ldr	x8, [x8, #0x90]
100000e64: 91004108    	add	x8, x8, #0x10
100000e68: f9000008    	str	x8, [x0]
100000e6c: f94003e8    	ldr	x8, [sp]
100000e70: f9000408    	str	x8, [x0, #0x8]
100000e74: 910043ff    	add	sp, sp, #0x10
100000e78: d65f03c0    	ret

0000000100000e7c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>:
100000e7c: d100c3ff    	sub	sp, sp, #0x30
100000e80: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e84: 910083fd    	add	x29, sp, #0x20
100000e88: f81f83a0    	stur	x0, [x29, #-0x8]
100000e8c: f9000be1    	str	x1, [sp, #0x10]
100000e90: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e94: f90007e0    	str	x0, [sp, #0x8]
100000e98: 94000005    	bl	0x100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000e9c: f94007e0    	ldr	x0, [sp, #0x8]
100000ea0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ea4: 9100c3ff    	add	sp, sp, #0x30
100000ea8: d65f03c0    	ret

0000000100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>:
100000eac: d10043ff    	sub	sp, sp, #0x10
100000eb0: f90007e0    	str	x0, [sp, #0x8]
100000eb4: f94007e0    	ldr	x0, [sp, #0x8]
100000eb8: 910043ff    	add	sp, sp, #0x10
100000ebc: d65f03c0    	ret

0000000100000ec0 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100000ec0: d10083ff    	sub	sp, sp, #0x20
100000ec4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ec8: 910043fd    	add	x29, sp, #0x10
100000ecc: f90007e0    	str	x0, [sp, #0x8]
100000ed0: f90003e1    	str	x1, [sp]
100000ed4: f94007e0    	ldr	x0, [sp, #0x8]
100000ed8: f94003e1    	ldr	x1, [sp]
100000edc: 94000004    	bl	0x100000eec <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000ee0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ee4: 910083ff    	add	sp, sp, #0x20
100000ee8: d65f03c0    	ret

0000000100000eec <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100000eec: d10043ff    	sub	sp, sp, #0x10
100000ef0: f90007e0    	str	x0, [sp, #0x8]
100000ef4: f90003e1    	str	x1, [sp]
100000ef8: f94007e0    	ldr	x0, [sp, #0x8]
100000efc: f94003e8    	ldr	x8, [sp]
100000f00: b9400108    	ldr	w8, [x8]
100000f04: b9000008    	str	w8, [x0]
100000f08: 910043ff    	add	sp, sp, #0x10
100000f0c: d65f03c0    	ret

0000000100000f10 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>:
100000f10: d10083ff    	sub	sp, sp, #0x20
100000f14: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f18: 910043fd    	add	x29, sp, #0x10
100000f1c: f90007e0    	str	x0, [sp, #0x8]
100000f20: f94007e0    	ldr	x0, [sp, #0x8]
100000f24: f90003e0    	str	x0, [sp]
100000f28: 97ffffe1    	bl	0x100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000f2c: f94003e0    	ldr	x0, [sp]
100000f30: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f34: 910083ff    	add	sp, sp, #0x20
100000f38: d65f03c0    	ret

0000000100000f3c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>:
100000f3c: d10083ff    	sub	sp, sp, #0x20
100000f40: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f44: 910043fd    	add	x29, sp, #0x10
100000f48: f90007e0    	str	x0, [sp, #0x8]
100000f4c: f94007e8    	ldr	x8, [sp, #0x8]
100000f50: f90003e8    	str	x8, [sp]
100000f54: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000f58: 91030129    	add	x9, x9, #0xc0
100000f5c: 91004129    	add	x9, x9, #0x10
100000f60: f9000109    	str	x9, [x8]
100000f64: 91006100    	add	x0, x8, #0x18
100000f68: 97ffff77    	bl	0x100000d44 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000f6c: f94003e0    	ldr	x0, [sp]
100000f70: 9400015e    	bl	0x1000014e8 <___stack_chk_guard+0x1000014e8>
100000f74: f94003e0    	ldr	x0, [sp]
100000f78: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f7c: 910083ff    	add	sp, sp, #0x20
100000f80: d65f03c0    	ret

0000000100000f84 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv>:
100000f84: d100c3ff    	sub	sp, sp, #0x30
100000f88: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000f8c: 910083fd    	add	x29, sp, #0x20
100000f90: f81f83a0    	stur	x0, [x29, #-0x8]
100000f94: f85f83a0    	ldur	x0, [x29, #-0x8]
100000f98: f90007e0    	str	x0, [sp, #0x8]
100000f9c: 97ffff54    	bl	0x100000cec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000fa0: f94007e0    	ldr	x0, [sp, #0x8]
100000fa4: 97fffe49    	bl	0x1000008c8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000fa8: aa0003e1    	mov	x1, x0
100000fac: d10027a0    	sub	x0, x29, #0x9
100000fb0: 94000157    	bl	0x10000150c <___stack_chk_guard+0x10000150c>
100000fb4: 14000001    	b	0x100000fb8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv+0x34>
100000fb8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000fbc: 9100c3ff    	add	sp, sp, #0x30
100000fc0: d65f03c0    	ret
100000fc4: 9400000b    	bl	0x100000ff0 <___clang_call_terminate>

0000000100000fc8 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE7destroyB8ne200100IivLi0EEEvRS2_PT_>:
100000fc8: d10083ff    	sub	sp, sp, #0x20
100000fcc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000fd0: 910043fd    	add	x29, sp, #0x10
100000fd4: f90007e0    	str	x0, [sp, #0x8]
100000fd8: f90003e1    	str	x1, [sp]
100000fdc: f94003e0    	ldr	x0, [sp]
100000fe0: 94000008    	bl	0x100001000 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>
100000fe4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000fe8: 910083ff    	add	sp, sp, #0x20
100000fec: d65f03c0    	ret

0000000100000ff0 <___clang_call_terminate>:
100000ff0: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000ff4: 910003fd    	mov	x29, sp
100000ff8: 94000148    	bl	0x100001518 <___stack_chk_guard+0x100001518>
100000ffc: 9400014a    	bl	0x100001524 <___stack_chk_guard+0x100001524>

0000000100001000 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>:
100001000: d10043ff    	sub	sp, sp, #0x10
100001004: f90007e0    	str	x0, [sp, #0x8]
100001008: 910043ff    	add	sp, sp, #0x10
10000100c: d65f03c0    	ret

0000000100001010 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>:
100001010: d100c3ff    	sub	sp, sp, #0x30
100001014: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001018: 910083fd    	add	x29, sp, #0x20
10000101c: f81f83a0    	stur	x0, [x29, #-0x8]
100001020: f9000be1    	str	x1, [sp, #0x10]
100001024: f90007e2    	str	x2, [sp, #0x8]
100001028: f85f83a0    	ldur	x0, [x29, #-0x8]
10000102c: f9400be1    	ldr	x1, [sp, #0x10]
100001030: f94007e2    	ldr	x2, [sp, #0x8]
100001034: 94000009    	bl	0x100001058 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>
100001038: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000103c: 9100c3ff    	add	sp, sp, #0x30
100001040: d65f03c0    	ret

0000000100001044 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>:
100001044: d10043ff    	sub	sp, sp, #0x10
100001048: f90007e0    	str	x0, [sp, #0x8]
10000104c: f94007e0    	ldr	x0, [sp, #0x8]
100001050: 910043ff    	add	sp, sp, #0x10
100001054: d65f03c0    	ret

0000000100001058 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>:
100001058: d100c3ff    	sub	sp, sp, #0x30
10000105c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001060: 910083fd    	add	x29, sp, #0x20
100001064: f81f83a0    	stur	x0, [x29, #-0x8]
100001068: f9000be1    	str	x1, [sp, #0x10]
10000106c: f90007e2    	str	x2, [sp, #0x8]
100001070: f9400be0    	ldr	x0, [sp, #0x10]
100001074: f94007e1    	ldr	x1, [sp, #0x8]
100001078: d2800102    	mov	x2, #0x8                ; =8
10000107c: 94000004    	bl	0x10000108c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
100001080: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001084: 9100c3ff    	add	sp, sp, #0x30
100001088: d65f03c0    	ret

000000010000108c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
10000108c: d10103ff    	sub	sp, sp, #0x40
100001090: a9037bfd    	stp	x29, x30, [sp, #0x30]
100001094: 9100c3fd    	add	x29, sp, #0x30
100001098: f81f83a0    	stur	x0, [x29, #-0x8]
10000109c: f81f03a1    	stur	x1, [x29, #-0x10]
1000010a0: f9000fe2    	str	x2, [sp, #0x18]
1000010a4: f85f03a8    	ldur	x8, [x29, #-0x10]
1000010a8: d37be908    	lsl	x8, x8, #5
1000010ac: f9000be8    	str	x8, [sp, #0x10]
1000010b0: f9400fe0    	ldr	x0, [sp, #0x18]
1000010b4: 97fffea4    	bl	0x100000b44 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
1000010b8: 36000100    	tbz	w0, #0x0, 0x1000010d8 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x4c>
1000010bc: 14000001    	b	0x1000010c0 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x34>
1000010c0: f9400fe8    	ldr	x8, [sp, #0x18]
1000010c4: f90007e8    	str	x8, [sp, #0x8]
1000010c8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000010cc: f94007e1    	ldr	x1, [sp, #0x8]
1000010d0: 94000008    	bl	0x1000010f0 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>
1000010d4: 14000004    	b	0x1000010e4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
1000010d8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000010dc: 94000010    	bl	0x10000111c <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>
1000010e0: 14000001    	b	0x1000010e4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
1000010e4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000010e8: 910103ff    	add	sp, sp, #0x40
1000010ec: d65f03c0    	ret

00000001000010f0 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>:
1000010f0: d10083ff    	sub	sp, sp, #0x20
1000010f4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000010f8: 910043fd    	add	x29, sp, #0x10
1000010fc: f90007e0    	str	x0, [sp, #0x8]
100001100: f90003e1    	str	x1, [sp]
100001104: f94007e0    	ldr	x0, [sp, #0x8]
100001108: f94003e1    	ldr	x1, [sp]
10000110c: 94000109    	bl	0x100001530 <___stack_chk_guard+0x100001530>
100001110: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001114: 910083ff    	add	sp, sp, #0x20
100001118: d65f03c0    	ret

000000010000111c <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>:
10000111c: d10083ff    	sub	sp, sp, #0x20
100001120: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001124: 910043fd    	add	x29, sp, #0x10
100001128: f90007e0    	str	x0, [sp, #0x8]
10000112c: f94007e0    	ldr	x0, [sp, #0x8]
100001130: 940000f1    	bl	0x1000014f4 <___stack_chk_guard+0x1000014f4>
100001134: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001138: 910083ff    	add	sp, sp, #0x20
10000113c: d65f03c0    	ret

0000000100001140 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>:
100001140: d10083ff    	sub	sp, sp, #0x20
100001144: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001148: 910043fd    	add	x29, sp, #0x10
10000114c: f90007e0    	str	x0, [sp, #0x8]
100001150: f94007e0    	ldr	x0, [sp, #0x8]
100001154: f90003e0    	str	x0, [sp]
100001158: 94000009    	bl	0x10000117c <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>
10000115c: f94003e0    	ldr	x0, [sp]
100001160: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001164: 910083ff    	add	sp, sp, #0x20
100001168: d65f03c0    	ret

000000010000116c <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>:
10000116c: d10043ff    	sub	sp, sp, #0x10
100001170: f90007e0    	str	x0, [sp, #0x8]
100001174: 910043ff    	add	sp, sp, #0x10
100001178: d65f03c0    	ret

000000010000117c <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>:
10000117c: d10043ff    	sub	sp, sp, #0x10
100001180: f90007e0    	str	x0, [sp, #0x8]
100001184: f94007e0    	ldr	x0, [sp, #0x8]
100001188: f900001f    	str	xzr, [x0]
10000118c: f900041f    	str	xzr, [x0, #0x8]
100001190: 910043ff    	add	sp, sp, #0x10
100001194: d65f03c0    	ret

0000000100001198 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>:
100001198: d10043ff    	sub	sp, sp, #0x10
10000119c: f90007e0    	str	x0, [sp, #0x8]
1000011a0: f94007e0    	ldr	x0, [sp, #0x8]
1000011a4: 910043ff    	add	sp, sp, #0x10
1000011a8: d65f03c0    	ret

00000001000011ac <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>:
1000011ac: d10083ff    	sub	sp, sp, #0x20
1000011b0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000011b4: 910043fd    	add	x29, sp, #0x10
1000011b8: f90007e0    	str	x0, [sp, #0x8]
1000011bc: f94007e0    	ldr	x0, [sp, #0x8]
1000011c0: f90003e0    	str	x0, [sp]
1000011c4: 94000005    	bl	0x1000011d8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>
1000011c8: f94003e0    	ldr	x0, [sp]
1000011cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000011d0: 910083ff    	add	sp, sp, #0x20
1000011d4: d65f03c0    	ret

00000001000011d8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>:
1000011d8: d10083ff    	sub	sp, sp, #0x20
1000011dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000011e0: 910043fd    	add	x29, sp, #0x10
1000011e4: f90007e0    	str	x0, [sp, #0x8]
1000011e8: f94007e8    	ldr	x8, [sp, #0x8]
1000011ec: f90003e8    	str	x8, [sp]
1000011f0: f9400908    	ldr	x8, [x8, #0x10]
1000011f4: b40000e8    	cbz	x8, 0x100001210 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
1000011f8: 14000001    	b	0x1000011fc <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x24>
1000011fc: f94003e0    	ldr	x0, [sp]
100001200: f9400801    	ldr	x1, [x0, #0x10]
100001204: f9400402    	ldr	x2, [x0, #0x8]
100001208: 97ffff82    	bl	0x100001010 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
10000120c: 14000001    	b	0x100001210 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
100001210: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001214: 910083ff    	add	sp, sp, #0x20
100001218: d65f03c0    	ret

000000010000121c <__ZNSt3__19allocatorIiEC2B8ne200100Ev>:
10000121c: d10083ff    	sub	sp, sp, #0x20
100001220: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001224: 910043fd    	add	x29, sp, #0x10
100001228: f90007e0    	str	x0, [sp, #0x8]
10000122c: f94007e0    	ldr	x0, [sp, #0x8]
100001230: f90003e0    	str	x0, [sp]
100001234: 94000005    	bl	0x100001248 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
100001238: f94003e0    	ldr	x0, [sp]
10000123c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001240: 910083ff    	add	sp, sp, #0x20
100001244: d65f03c0    	ret

0000000100001248 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
100001248: d10043ff    	sub	sp, sp, #0x10
10000124c: f90007e0    	str	x0, [sp, #0x8]
100001250: f94007e0    	ldr	x0, [sp, #0x8]
100001254: 910043ff    	add	sp, sp, #0x10
100001258: d65f03c0    	ret

000000010000125c <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>:
10000125c: d100c3ff    	sub	sp, sp, #0x30
100001260: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001264: 910083fd    	add	x29, sp, #0x20
100001268: f9000be0    	str	x0, [sp, #0x10]
10000126c: f9400be8    	ldr	x8, [sp, #0x10]
100001270: f90007e8    	str	x8, [sp, #0x8]
100001274: aa0803e9    	mov	x9, x8
100001278: f81f83a9    	stur	x9, [x29, #-0x8]
10000127c: f9400508    	ldr	x8, [x8, #0x8]
100001280: b40000c8    	cbz	x8, 0x100001298 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
100001284: 14000001    	b	0x100001288 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x2c>
100001288: f94007e8    	ldr	x8, [sp, #0x8]
10000128c: f9400500    	ldr	x0, [x8, #0x8]
100001290: 94000006    	bl	0x1000012a8 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
100001294: 14000001    	b	0x100001298 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
100001298: f85f83a0    	ldur	x0, [x29, #-0x8]
10000129c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000012a0: 9100c3ff    	add	sp, sp, #0x30
1000012a4: d65f03c0    	ret

00000001000012a8 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
1000012a8: d10083ff    	sub	sp, sp, #0x20
1000012ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012b0: 910043fd    	add	x29, sp, #0x10
1000012b4: f90007e0    	str	x0, [sp, #0x8]
1000012b8: f94007e0    	ldr	x0, [sp, #0x8]
1000012bc: f90003e0    	str	x0, [sp]
1000012c0: 94000009    	bl	0x1000012e4 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
1000012c4: 360000a0    	tbz	w0, #0x0, 0x1000012d8 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
1000012c8: 14000001    	b	0x1000012cc <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
1000012cc: f94003e0    	ldr	x0, [sp]
1000012d0: 9400009b    	bl	0x10000153c <___stack_chk_guard+0x10000153c>
1000012d4: 14000001    	b	0x1000012d8 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
1000012d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012dc: 910083ff    	add	sp, sp, #0x20
1000012e0: d65f03c0    	ret

00000001000012e4 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
1000012e4: d100c3ff    	sub	sp, sp, #0x30
1000012e8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000012ec: 910083fd    	add	x29, sp, #0x20
1000012f0: f9000be0    	str	x0, [sp, #0x10]
1000012f4: f9400be8    	ldr	x8, [sp, #0x10]
1000012f8: f90007e8    	str	x8, [sp, #0x8]
1000012fc: 91002100    	add	x0, x8, #0x8
100001300: 94000017    	bl	0x10000135c <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
100001304: b1000408    	adds	x8, x0, #0x1
100001308: 54000161    	b.ne	0x100001334 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
10000130c: 14000001    	b	0x100001310 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
100001310: f94007e0    	ldr	x0, [sp, #0x8]
100001314: f9400008    	ldr	x8, [x0]
100001318: f9400908    	ldr	x8, [x8, #0x10]
10000131c: d63f0100    	blr	x8
100001320: 52800028    	mov	w8, #0x1                ; =1
100001324: 12000108    	and	w8, w8, #0x1
100001328: 12000108    	and	w8, w8, #0x1
10000132c: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001330: 14000006    	b	0x100001348 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100001334: 52800008    	mov	w8, #0x0                ; =0
100001338: 12000108    	and	w8, w8, #0x1
10000133c: 12000108    	and	w8, w8, #0x1
100001340: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001344: 14000001    	b	0x100001348 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100001348: 385ff3a8    	ldurb	w8, [x29, #-0x1]
10000134c: 12000100    	and	w0, w8, #0x1
100001350: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001354: 9100c3ff    	add	sp, sp, #0x30
100001358: d65f03c0    	ret

000000010000135c <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
10000135c: d10083ff    	sub	sp, sp, #0x20
100001360: f9000fe0    	str	x0, [sp, #0x18]
100001364: f9400fe8    	ldr	x8, [sp, #0x18]
100001368: 92800009    	mov	x9, #-0x1               ; =-1
10000136c: f9000be9    	str	x9, [sp, #0x10]
100001370: f9400be9    	ldr	x9, [sp, #0x10]
100001374: f8e90108    	ldaddal	x9, x8, [x8]
100001378: 8b090108    	add	x8, x8, x9
10000137c: f90007e8    	str	x8, [sp, #0x8]
100001380: f94007e0    	ldr	x0, [sp, #0x8]
100001384: 910083ff    	add	sp, sp, #0x20
100001388: d65f03c0    	ret

000000010000138c <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_>:
10000138c: d100c3ff    	sub	sp, sp, #0x30
100001390: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001394: 910083fd    	add	x29, sp, #0x20
100001398: f9000be0    	str	x0, [sp, #0x10]
10000139c: f90007e1    	str	x1, [sp, #0x8]
1000013a0: f9400be8    	ldr	x8, [sp, #0x10]
1000013a4: f90003e8    	str	x8, [sp]
1000013a8: aa0803e9    	mov	x9, x8
1000013ac: f81f83a9    	stur	x9, [x29, #-0x8]
1000013b0: f94007e9    	ldr	x9, [sp, #0x8]
1000013b4: f9400129    	ldr	x9, [x9]
1000013b8: f9000109    	str	x9, [x8]
1000013bc: f94007e9    	ldr	x9, [sp, #0x8]
1000013c0: f9400529    	ldr	x9, [x9, #0x8]
1000013c4: f9000509    	str	x9, [x8, #0x8]
1000013c8: f9400508    	ldr	x8, [x8, #0x8]
1000013cc: b40000c8    	cbz	x8, 0x1000013e4 <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_+0x58>
1000013d0: 14000001    	b	0x1000013d4 <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_+0x48>
1000013d4: f94003e8    	ldr	x8, [sp]
1000013d8: f9400500    	ldr	x0, [x8, #0x8]
1000013dc: 94000006    	bl	0x1000013f4 <__ZNSt3__119__shared_weak_count12__add_sharedB8ne200100Ev>
1000013e0: 14000001    	b	0x1000013e4 <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_+0x58>
1000013e4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000013e8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000013ec: 9100c3ff    	add	sp, sp, #0x30
1000013f0: d65f03c0    	ret

00000001000013f4 <__ZNSt3__119__shared_weak_count12__add_sharedB8ne200100Ev>:
1000013f4: d10083ff    	sub	sp, sp, #0x20
1000013f8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000013fc: 910043fd    	add	x29, sp, #0x10
100001400: f90007e0    	str	x0, [sp, #0x8]
100001404: f94007e0    	ldr	x0, [sp, #0x8]
100001408: 94000004    	bl	0x100001418 <__ZNSt3__114__shared_count12__add_sharedB8ne200100Ev>
10000140c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001410: 910083ff    	add	sp, sp, #0x20
100001414: d65f03c0    	ret

0000000100001418 <__ZNSt3__114__shared_count12__add_sharedB8ne200100Ev>:
100001418: d10083ff    	sub	sp, sp, #0x20
10000141c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001420: 910043fd    	add	x29, sp, #0x10
100001424: f90007e0    	str	x0, [sp, #0x8]
100001428: f94007e8    	ldr	x8, [sp, #0x8]
10000142c: 91002100    	add	x0, x8, #0x8
100001430: 94000004    	bl	0x100001440 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>
100001434: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001438: 910083ff    	add	sp, sp, #0x20
10000143c: d65f03c0    	ret

0000000100001440 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>:
100001440: d10083ff    	sub	sp, sp, #0x20
100001444: f9000fe0    	str	x0, [sp, #0x18]
100001448: f9400fe8    	ldr	x8, [sp, #0x18]
10000144c: d2800029    	mov	x9, #0x1                ; =1
100001450: f9000be9    	str	x9, [sp, #0x10]
100001454: f9400be9    	ldr	x9, [sp, #0x10]
100001458: f8290108    	ldadd	x9, x8, [x8]
10000145c: 8b090108    	add	x8, x8, x9
100001460: f90007e8    	str	x8, [sp, #0x8]
100001464: f94007e0    	ldr	x0, [sp, #0x8]
100001468: 910083ff    	add	sp, sp, #0x20
10000146c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001470 <__stubs>:
100001470: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001474: f9400a10    	ldr	x16, [x16, #0x10]
100001478: d61f0200    	br	x16
10000147c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001480: f9400e10    	ldr	x16, [x16, #0x18]
100001484: d61f0200    	br	x16
100001488: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000148c: f9401210    	ldr	x16, [x16, #0x20]
100001490: d61f0200    	br	x16
100001494: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001498: f9401610    	ldr	x16, [x16, #0x28]
10000149c: d61f0200    	br	x16
1000014a0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014a4: f9401a10    	ldr	x16, [x16, #0x30]
1000014a8: d61f0200    	br	x16
1000014ac: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014b0: f9401e10    	ldr	x16, [x16, #0x38]
1000014b4: d61f0200    	br	x16
1000014b8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014bc: f9402a10    	ldr	x16, [x16, #0x50]
1000014c0: d61f0200    	br	x16
1000014c4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014c8: f9402e10    	ldr	x16, [x16, #0x58]
1000014cc: d61f0200    	br	x16
1000014d0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014d4: f9403210    	ldr	x16, [x16, #0x60]
1000014d8: d61f0200    	br	x16
1000014dc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014e0: f9403610    	ldr	x16, [x16, #0x68]
1000014e4: d61f0200    	br	x16
1000014e8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014ec: f9403a10    	ldr	x16, [x16, #0x70]
1000014f0: d61f0200    	br	x16
1000014f4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000014f8: f9404210    	ldr	x16, [x16, #0x80]
1000014fc: d61f0200    	br	x16
100001500: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001504: f9404610    	ldr	x16, [x16, #0x88]
100001508: d61f0200    	br	x16
10000150c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001510: f9404e10    	ldr	x16, [x16, #0x98]
100001514: d61f0200    	br	x16
100001518: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000151c: f9405210    	ldr	x16, [x16, #0xa0]
100001520: d61f0200    	br	x16
100001524: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001528: f9405610    	ldr	x16, [x16, #0xa8]
10000152c: d61f0200    	br	x16
100001530: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001534: f9405a10    	ldr	x16, [x16, #0xb0]
100001538: d61f0200    	br	x16
10000153c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001540: f9405e10    	ldr	x16, [x16, #0xb8]
100001544: d61f0200    	br	x16
