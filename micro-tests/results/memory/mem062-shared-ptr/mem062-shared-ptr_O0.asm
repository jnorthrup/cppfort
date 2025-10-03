
/Users/jim/work/cppfort/micro-tests/results/memory/mem062-shared-ptr/mem062-shared-ptr_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z15test_shared_ptrv>:
100000538: d100c3ff    	sub	sp, sp, #0x30
10000053c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000540: 910083fd    	add	x29, sp, #0x20
100000544: 910033e0    	add	x0, sp, #0xc
100000548: 52800548    	mov	w8, #0x2a               ; =42
10000054c: b9000fe8    	str	w8, [sp, #0xc]
100000550: 910043e8    	add	x8, sp, #0x10
100000554: f90003e8    	str	x8, [sp]
100000558: 9400000c    	bl	0x100000588 <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>
10000055c: f94003e0    	ldr	x0, [sp]
100000560: 9400001a    	bl	0x1000005c8 <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>
100000564: aa0003e8    	mov	x8, x0
100000568: f94003e0    	ldr	x0, [sp]
10000056c: b9400108    	ldr	w8, [x8]
100000570: b9000be8    	str	w8, [sp, #0x8]
100000574: 9400001b    	bl	0x1000005e0 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000578: b9400be0    	ldr	w0, [sp, #0x8]
10000057c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000580: 9100c3ff    	add	sp, sp, #0x30
100000584: d65f03c0    	ret

0000000100000588 <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>:
100000588: d10103ff    	sub	sp, sp, #0x40
10000058c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000590: 9100c3fd    	add	x29, sp, #0x30
100000594: f9000be8    	str	x8, [sp, #0x10]
100000598: f81f83a8    	stur	x8, [x29, #-0x8]
10000059c: f81f03a0    	stur	x0, [x29, #-0x10]
1000005a0: d10047a0    	sub	x0, x29, #0x11
1000005a4: f90007e0    	str	x0, [sp, #0x8]
1000005a8: 94000058    	bl	0x100000708 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>
1000005ac: f94007e0    	ldr	x0, [sp, #0x8]
1000005b0: f9400be8    	ldr	x8, [sp, #0x10]
1000005b4: f85f03a1    	ldur	x1, [x29, #-0x10]
1000005b8: 9400001d    	bl	0x10000062c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>
1000005bc: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000005c0: 910103ff    	add	sp, sp, #0x40
1000005c4: d65f03c0    	ret

00000001000005c8 <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>:
1000005c8: d10043ff    	sub	sp, sp, #0x10
1000005cc: f90007e0    	str	x0, [sp, #0x8]
1000005d0: f94007e8    	ldr	x8, [sp, #0x8]
1000005d4: f9400100    	ldr	x0, [x8]
1000005d8: 910043ff    	add	sp, sp, #0x10
1000005dc: d65f03c0    	ret

00000001000005e0 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>:
1000005e0: d10083ff    	sub	sp, sp, #0x20
1000005e4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000005e8: 910043fd    	add	x29, sp, #0x10
1000005ec: f90007e0    	str	x0, [sp, #0x8]
1000005f0: f94007e0    	ldr	x0, [sp, #0x8]
1000005f4: f90003e0    	str	x0, [sp]
1000005f8: 94000306    	bl	0x100001210 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>
1000005fc: f94003e0    	ldr	x0, [sp]
100000600: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000604: 910083ff    	add	sp, sp, #0x20
100000608: d65f03c0    	ret

000000010000060c <_main>:
10000060c: d10083ff    	sub	sp, sp, #0x20
100000610: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000614: 910043fd    	add	x29, sp, #0x10
100000618: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000061c: 97ffffc7    	bl	0x100000538 <__Z15test_shared_ptrv>
100000620: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000624: 910083ff    	add	sp, sp, #0x20
100000628: d65f03c0    	ret

000000010000062c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>:
10000062c: d10203ff    	sub	sp, sp, #0x80
100000630: a9077bfd    	stp	x29, x30, [sp, #0x70]
100000634: 9101c3fd    	add	x29, sp, #0x70
100000638: f9000be8    	str	x8, [sp, #0x10]
10000063c: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000640: f9400529    	ldr	x9, [x9, #0x8]
100000644: f9400129    	ldr	x9, [x9]
100000648: f81f83a9    	stur	x9, [x29, #-0x8]
10000064c: f81d83a8    	stur	x8, [x29, #-0x28]
100000650: f81d03a0    	stur	x0, [x29, #-0x30]
100000654: f9001fe1    	str	x1, [sp, #0x38]
100000658: d10083a0    	sub	x0, x29, #0x20
10000065c: d2800021    	mov	x1, #0x1                ; =1
100000660: 94000035    	bl	0x100000734 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>
100000664: 14000001    	b	0x100000668 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x3c>
100000668: d10083a0    	sub	x0, x29, #0x20
10000066c: 9400003f    	bl	0x100000768 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>
100000670: f9401fe1    	ldr	x1, [sp, #0x38]
100000674: 94000043    	bl	0x100000780 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>
100000678: 14000001    	b	0x10000067c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x50>
10000067c: d10083a0    	sub	x0, x29, #0x20
100000680: f90007e0    	str	x0, [sp, #0x8]
100000684: 9400004c    	bl	0x1000007b4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>
100000688: f9000fe0    	str	x0, [sp, #0x18]
10000068c: f9400fe0    	ldr	x0, [sp, #0x18]
100000690: 9400007b    	bl	0x10000087c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000694: f9400be8    	ldr	x8, [sp, #0x10]
100000698: f9400fe1    	ldr	x1, [sp, #0x18]
10000069c: 94000329    	bl	0x100001340 <___stack_chk_guard+0x100001340>
1000006a0: f94007e0    	ldr	x0, [sp, #0x8]
1000006a4: 94000080    	bl	0x1000008a4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
1000006a8: f85f83a9    	ldur	x9, [x29, #-0x8]
1000006ac: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000006b0: f9400508    	ldr	x8, [x8, #0x8]
1000006b4: f9400108    	ldr	x8, [x8]
1000006b8: eb090108    	subs	x8, x8, x9
1000006bc: 54000060    	b.eq	0x1000006c8 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x9c>
1000006c0: 14000001    	b	0x1000006c4 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x98>
1000006c4: 94000322    	bl	0x10000134c <___stack_chk_guard+0x10000134c>
1000006c8: a9477bfd    	ldp	x29, x30, [sp, #0x70]
1000006cc: 910203ff    	add	sp, sp, #0x80
1000006d0: d65f03c0    	ret
1000006d4: f90017e0    	str	x0, [sp, #0x28]
1000006d8: aa0103e8    	mov	x8, x1
1000006dc: b90027e8    	str	w8, [sp, #0x24]
1000006e0: d10083a0    	sub	x0, x29, #0x20
1000006e4: 94000070    	bl	0x1000008a4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
1000006e8: 14000001    	b	0x1000006ec <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xc0>
1000006ec: f94017e0    	ldr	x0, [sp, #0x28]
1000006f0: f90003e0    	str	x0, [sp]
1000006f4: 14000003    	b	0x100000700 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
1000006f8: f90003e0    	str	x0, [sp]
1000006fc: 14000001    	b	0x100000700 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
100000700: f94003e0    	ldr	x0, [sp]
100000704: 94000315    	bl	0x100001358 <___stack_chk_guard+0x100001358>

0000000100000708 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>:
100000708: d10083ff    	sub	sp, sp, #0x20
10000070c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000710: 910043fd    	add	x29, sp, #0x10
100000714: f90007e0    	str	x0, [sp, #0x8]
100000718: f94007e0    	ldr	x0, [sp, #0x8]
10000071c: f90003e0    	str	x0, [sp]
100000720: 940002ac    	bl	0x1000011d0 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>
100000724: f94003e0    	ldr	x0, [sp]
100000728: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000072c: 910083ff    	add	sp, sp, #0x20
100000730: d65f03c0    	ret

0000000100000734 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>:
100000734: d100c3ff    	sub	sp, sp, #0x30
100000738: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000073c: 910083fd    	add	x29, sp, #0x20
100000740: f9000be0    	str	x0, [sp, #0x10]
100000744: f90007e1    	str	x1, [sp, #0x8]
100000748: f9400be0    	ldr	x0, [sp, #0x10]
10000074c: f90003e0    	str	x0, [sp]
100000750: f94007e1    	ldr	x1, [sp, #0x8]
100000754: 9400005f    	bl	0x1000008d0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>
100000758: f94003e0    	ldr	x0, [sp]
10000075c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000760: 9100c3ff    	add	sp, sp, #0x30
100000764: d65f03c0    	ret

0000000100000768 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>:
100000768: d10043ff    	sub	sp, sp, #0x10
10000076c: f90007e0    	str	x0, [sp, #0x8]
100000770: f94007e8    	ldr	x8, [sp, #0x8]
100000774: f9400900    	ldr	x0, [x8, #0x10]
100000778: 910043ff    	add	sp, sp, #0x10
10000077c: d65f03c0    	ret

0000000100000780 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000780: d100c3ff    	sub	sp, sp, #0x30
100000784: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000788: 910083fd    	add	x29, sp, #0x20
10000078c: f9000be0    	str	x0, [sp, #0x10]
100000790: f90007e1    	str	x1, [sp, #0x8]
100000794: f9400be0    	ldr	x0, [sp, #0x10]
100000798: f90003e0    	str	x0, [sp]
10000079c: f94007e1    	ldr	x1, [sp, #0x8]
1000007a0: 940000f1    	bl	0x100000b64 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>
1000007a4: f94003e0    	ldr	x0, [sp]
1000007a8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000007ac: 9100c3ff    	add	sp, sp, #0x30
1000007b0: d65f03c0    	ret

00000001000007b4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>:
1000007b4: d10043ff    	sub	sp, sp, #0x10
1000007b8: f90007e0    	str	x0, [sp, #0x8]
1000007bc: f94007e8    	ldr	x8, [sp, #0x8]
1000007c0: f9400909    	ldr	x9, [x8, #0x10]
1000007c4: f90003e9    	str	x9, [sp]
1000007c8: f900091f    	str	xzr, [x8, #0x10]
1000007cc: f94003e0    	ldr	x0, [sp]
1000007d0: 910043ff    	add	sp, sp, #0x10
1000007d4: d65f03c0    	ret

00000001000007d8 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_>:
1000007d8: d10143ff    	sub	sp, sp, #0x50
1000007dc: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000007e0: 910103fd    	add	x29, sp, #0x40
1000007e4: f9000fe8    	str	x8, [sp, #0x18]
1000007e8: aa0003e8    	mov	x8, x0
1000007ec: f9400fe0    	ldr	x0, [sp, #0x18]
1000007f0: aa0003e9    	mov	x9, x0
1000007f4: f81f83a9    	stur	x9, [x29, #-0x8]
1000007f8: f81f03a8    	stur	x8, [x29, #-0x10]
1000007fc: f81e83a1    	stur	x1, [x29, #-0x18]
100000800: 52800008    	mov	w8, #0x0                ; =0
100000804: 52800029    	mov	w9, #0x1                ; =1
100000808: b90023e9    	str	w9, [sp, #0x20]
10000080c: 12000108    	and	w8, w8, #0x1
100000810: 12000108    	and	w8, w8, #0x1
100000814: 381e73a8    	sturb	w8, [x29, #-0x19]
100000818: 94000237    	bl	0x1000010f4 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>
10000081c: f9400fe0    	ldr	x0, [sp, #0x18]
100000820: f85f03a8    	ldur	x8, [x29, #-0x10]
100000824: f9000008    	str	x8, [x0]
100000828: f85e83a8    	ldur	x8, [x29, #-0x18]
10000082c: f9000408    	str	x8, [x0, #0x8]
100000830: f940000a    	ldr	x10, [x0]
100000834: f9400008    	ldr	x8, [x0]
100000838: 910003e9    	mov	x9, sp
10000083c: f900012a    	str	x10, [x9]
100000840: f9000528    	str	x8, [x9, #0x8]
100000844: 94000237    	bl	0x100001120 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>
100000848: b94023e9    	ldr	w9, [sp, #0x20]
10000084c: 12000128    	and	w8, w9, #0x1
100000850: 0a090108    	and	w8, w8, w9
100000854: 381e73a8    	sturb	w8, [x29, #-0x19]
100000858: 385e73a8    	ldurb	w8, [x29, #-0x19]
10000085c: 370000a8    	tbnz	w8, #0x0, 0x100000870 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
100000860: 14000001    	b	0x100000864 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x8c>
100000864: f9400fe0    	ldr	x0, [sp, #0x18]
100000868: 97ffff5e    	bl	0x1000005e0 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
10000086c: 14000001    	b	0x100000870 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
100000870: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000874: 910143ff    	add	sp, sp, #0x50
100000878: d65f03c0    	ret

000000010000087c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>:
10000087c: d10083ff    	sub	sp, sp, #0x20
100000880: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000884: 910043fd    	add	x29, sp, #0x10
100000888: f90007e0    	str	x0, [sp, #0x8]
10000088c: f94007e8    	ldr	x8, [sp, #0x8]
100000890: 91006100    	add	x0, x8, #0x18
100000894: 9400022e    	bl	0x10000114c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>
100000898: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000089c: 910083ff    	add	sp, sp, #0x20
1000008a0: d65f03c0    	ret

00000001000008a4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>:
1000008a4: d10083ff    	sub	sp, sp, #0x20
1000008a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000008ac: 910043fd    	add	x29, sp, #0x10
1000008b0: f90007e0    	str	x0, [sp, #0x8]
1000008b4: f94007e0    	ldr	x0, [sp, #0x8]
1000008b8: f90003e0    	str	x0, [sp]
1000008bc: 94000229    	bl	0x100001160 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>
1000008c0: f94003e0    	ldr	x0, [sp]
1000008c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000008c8: 910083ff    	add	sp, sp, #0x20
1000008cc: d65f03c0    	ret

00000001000008d0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>:
1000008d0: d100c3ff    	sub	sp, sp, #0x30
1000008d4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000008d8: 910083fd    	add	x29, sp, #0x20
1000008dc: f9000be0    	str	x0, [sp, #0x10]
1000008e0: f90007e1    	str	x1, [sp, #0x8]
1000008e4: f9400be0    	ldr	x0, [sp, #0x10]
1000008e8: f90003e0    	str	x0, [sp]
1000008ec: d10007a1    	sub	x1, x29, #0x1
1000008f0: 9400000c    	bl	0x100000920 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
1000008f4: f94003e0    	ldr	x0, [sp]
1000008f8: f94007e8    	ldr	x8, [sp, #0x8]
1000008fc: f9000408    	str	x8, [x0, #0x8]
100000900: f9400401    	ldr	x1, [x0, #0x8]
100000904: 94000014    	bl	0x100000954 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>
100000908: aa0003e8    	mov	x8, x0
10000090c: f94003e0    	ldr	x0, [sp]
100000910: f9000808    	str	x8, [x0, #0x10]
100000914: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000918: 9100c3ff    	add	sp, sp, #0x30
10000091c: d65f03c0    	ret

0000000100000920 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>:
100000920: d100c3ff    	sub	sp, sp, #0x30
100000924: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000928: 910083fd    	add	x29, sp, #0x20
10000092c: f81f83a0    	stur	x0, [x29, #-0x8]
100000930: f9000be1    	str	x1, [sp, #0x10]
100000934: f85f83a0    	ldur	x0, [x29, #-0x8]
100000938: f90007e0    	str	x0, [sp, #0x8]
10000093c: f9400be1    	ldr	x1, [sp, #0x10]
100000940: 94000010    	bl	0x100000980 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>
100000944: f94007e0    	ldr	x0, [sp, #0x8]
100000948: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000094c: 9100c3ff    	add	sp, sp, #0x30
100000950: d65f03c0    	ret

0000000100000954 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>:
100000954: d10083ff    	sub	sp, sp, #0x20
100000958: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000095c: 910043fd    	add	x29, sp, #0x10
100000960: f90007e0    	str	x0, [sp, #0x8]
100000964: f90003e1    	str	x1, [sp]
100000968: f94007e0    	ldr	x0, [sp, #0x8]
10000096c: f94003e1    	ldr	x1, [sp]
100000970: 94000015    	bl	0x1000009c4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>
100000974: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000978: 910083ff    	add	sp, sp, #0x20
10000097c: d65f03c0    	ret

0000000100000980 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>:
100000980: d100c3ff    	sub	sp, sp, #0x30
100000984: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000988: 910083fd    	add	x29, sp, #0x20
10000098c: f81f83a0    	stur	x0, [x29, #-0x8]
100000990: f9000be1    	str	x1, [sp, #0x10]
100000994: f85f83a0    	ldur	x0, [x29, #-0x8]
100000998: f90007e0    	str	x0, [sp, #0x8]
10000099c: 94000005    	bl	0x1000009b0 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>
1000009a0: f94007e0    	ldr	x0, [sp, #0x8]
1000009a4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000009a8: 9100c3ff    	add	sp, sp, #0x30
1000009ac: d65f03c0    	ret

00000001000009b0 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>:
1000009b0: d10043ff    	sub	sp, sp, #0x10
1000009b4: f90007e0    	str	x0, [sp, #0x8]
1000009b8: f94007e0    	ldr	x0, [sp, #0x8]
1000009bc: 910043ff    	add	sp, sp, #0x10
1000009c0: d65f03c0    	ret

00000001000009c4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>:
1000009c4: d100c3ff    	sub	sp, sp, #0x30
1000009c8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000009cc: 910083fd    	add	x29, sp, #0x20
1000009d0: f81f83a0    	stur	x0, [x29, #-0x8]
1000009d4: f9000be1    	str	x1, [sp, #0x10]
1000009d8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000009dc: f9400be8    	ldr	x8, [sp, #0x10]
1000009e0: f90007e8    	str	x8, [sp, #0x8]
1000009e4: 94000260    	bl	0x100001364 <___stack_chk_guard+0x100001364>
1000009e8: f94007e8    	ldr	x8, [sp, #0x8]
1000009ec: eb000108    	subs	x8, x8, x0
1000009f0: 54000069    	b.ls	0x1000009fc <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x38>
1000009f4: 14000001    	b	0x1000009f8 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x34>
1000009f8: 94000011    	bl	0x100000a3c <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
1000009fc: f9400be0    	ldr	x0, [sp, #0x10]
100000a00: d2800101    	mov	x1, #0x8                ; =8
100000a04: 9400001b    	bl	0x100000a70 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>
100000a08: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a0c: 9100c3ff    	add	sp, sp, #0x30
100000a10: d65f03c0    	ret

0000000100000a14 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8max_sizeB8ne200100IS5_vLi0EEEmRKS5_>:
100000a14: d10083ff    	sub	sp, sp, #0x20
100000a18: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a1c: 910043fd    	add	x29, sp, #0x10
100000a20: f90007e0    	str	x0, [sp, #0x8]
100000a24: 9400002e    	bl	0x100000adc <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000a28: d2800408    	mov	x8, #0x20               ; =32
100000a2c: 9ac80800    	udiv	x0, x0, x8
100000a30: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a34: 910083ff    	add	sp, sp, #0x20
100000a38: d65f03c0    	ret

0000000100000a3c <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000a3c: d10083ff    	sub	sp, sp, #0x20
100000a40: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a44: 910043fd    	add	x29, sp, #0x10
100000a48: d2800100    	mov	x0, #0x8                ; =8
100000a4c: 94000249    	bl	0x100001370 <___stack_chk_guard+0x100001370>
100000a50: f90007e0    	str	x0, [sp, #0x8]
100000a54: 9400024a    	bl	0x10000137c <___stack_chk_guard+0x10000137c>
100000a58: f94007e0    	ldr	x0, [sp, #0x8]
100000a5c: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000a60: f9402021    	ldr	x1, [x1, #0x40]
100000a64: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000a68: f9402442    	ldr	x2, [x2, #0x48]
100000a6c: 94000247    	bl	0x100001388 <___stack_chk_guard+0x100001388>

0000000100000a70 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>:
100000a70: d10103ff    	sub	sp, sp, #0x40
100000a74: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000a78: 9100c3fd    	add	x29, sp, #0x30
100000a7c: f81f03a0    	stur	x0, [x29, #-0x10]
100000a80: f9000fe1    	str	x1, [sp, #0x18]
100000a84: f85f03a8    	ldur	x8, [x29, #-0x10]
100000a88: d37be908    	lsl	x8, x8, #5
100000a8c: f9000be8    	str	x8, [sp, #0x10]
100000a90: f9400fe0    	ldr	x0, [sp, #0x18]
100000a94: 94000019    	bl	0x100000af8 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000a98: 36000120    	tbz	w0, #0x0, 0x100000abc <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x4c>
100000a9c: 14000001    	b	0x100000aa0 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x30>
100000aa0: f9400fe8    	ldr	x8, [sp, #0x18]
100000aa4: f90007e8    	str	x8, [sp, #0x8]
100000aa8: f9400be0    	ldr	x0, [sp, #0x10]
100000aac: f94007e1    	ldr	x1, [sp, #0x8]
100000ab0: 94000019    	bl	0x100000b14 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>
100000ab4: f81f83a0    	stur	x0, [x29, #-0x8]
100000ab8: 14000005    	b	0x100000acc <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000abc: f9400be0    	ldr	x0, [sp, #0x10]
100000ac0: 94000020    	bl	0x100000b40 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>
100000ac4: f81f83a0    	stur	x0, [x29, #-0x8]
100000ac8: 14000001    	b	0x100000acc <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000acc: f85f83a0    	ldur	x0, [x29, #-0x8]
100000ad0: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000ad4: 910103ff    	add	sp, sp, #0x40
100000ad8: d65f03c0    	ret

0000000100000adc <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000adc: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000ae0: 910003fd    	mov	x29, sp
100000ae4: 94000003    	bl	0x100000af0 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000ae8: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000aec: d65f03c0    	ret

0000000100000af0 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000af0: 92800000    	mov	x0, #-0x1               ; =-1
100000af4: d65f03c0    	ret

0000000100000af8 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000af8: d10043ff    	sub	sp, sp, #0x10
100000afc: f90007e0    	str	x0, [sp, #0x8]
100000b00: f94007e8    	ldr	x8, [sp, #0x8]
100000b04: f1004108    	subs	x8, x8, #0x10
100000b08: 1a9f97e0    	cset	w0, hi
100000b0c: 910043ff    	add	sp, sp, #0x10
100000b10: d65f03c0    	ret

0000000100000b14 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>:
100000b14: d10083ff    	sub	sp, sp, #0x20
100000b18: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000b1c: 910043fd    	add	x29, sp, #0x10
100000b20: f90007e0    	str	x0, [sp, #0x8]
100000b24: f90003e1    	str	x1, [sp]
100000b28: f94007e0    	ldr	x0, [sp, #0x8]
100000b2c: f94003e1    	ldr	x1, [sp]
100000b30: 94000219    	bl	0x100001394 <___stack_chk_guard+0x100001394>
100000b34: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000b38: 910083ff    	add	sp, sp, #0x20
100000b3c: d65f03c0    	ret

0000000100000b40 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>:
100000b40: d10083ff    	sub	sp, sp, #0x20
100000b44: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000b48: 910043fd    	add	x29, sp, #0x10
100000b4c: f90007e0    	str	x0, [sp, #0x8]
100000b50: f94007e0    	ldr	x0, [sp, #0x8]
100000b54: 94000213    	bl	0x1000013a0 <___stack_chk_guard+0x1000013a0>
100000b58: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000b5c: 910083ff    	add	sp, sp, #0x20
100000b60: d65f03c0    	ret

0000000100000b64 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000b64: d10103ff    	sub	sp, sp, #0x40
100000b68: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000b6c: 9100c3fd    	add	x29, sp, #0x30
100000b70: f81f03a0    	stur	x0, [x29, #-0x10]
100000b74: f9000fe1    	str	x1, [sp, #0x18]
100000b78: f85f03a0    	ldur	x0, [x29, #-0x10]
100000b7c: f90003e0    	str	x0, [sp]
100000b80: d2800001    	mov	x1, #0x0                ; =0
100000b84: 94000027    	bl	0x100000c20 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000b88: f94003e8    	ldr	x8, [sp]
100000b8c: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000b90: 91030129    	add	x9, x9, #0xc0
100000b94: 91004129    	add	x9, x9, #0x10
100000b98: f9000109    	str	x9, [x8]
100000b9c: 91006100    	add	x0, x8, #0x18
100000ba0: d10007a1    	sub	x1, x29, #0x1
100000ba4: 94000032    	bl	0x100000c6c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>
100000ba8: 14000001    	b	0x100000bac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x48>
100000bac: f94003e0    	ldr	x0, [sp]
100000bb0: 9400003c    	bl	0x100000ca0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000bb4: f94003e0    	ldr	x0, [sp]
100000bb8: 97ffff31    	bl	0x10000087c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000bbc: aa0003e1    	mov	x1, x0
100000bc0: f9400fe2    	ldr	x2, [sp, #0x18]
100000bc4: 91002fe0    	add	x0, sp, #0xb
100000bc8: 940001f9    	bl	0x1000013ac <___stack_chk_guard+0x1000013ac>
100000bcc: 14000001    	b	0x100000bd0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x6c>
100000bd0: f94003e0    	ldr	x0, [sp]
100000bd4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000bd8: 910103ff    	add	sp, sp, #0x40
100000bdc: d65f03c0    	ret
100000be0: f9000be0    	str	x0, [sp, #0x10]
100000be4: aa0103e8    	mov	x8, x1
100000be8: b9000fe8    	str	w8, [sp, #0xc]
100000bec: 14000008    	b	0x100000c0c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000bf0: f94003e8    	ldr	x8, [sp]
100000bf4: f9000be0    	str	x0, [sp, #0x10]
100000bf8: aa0103e9    	mov	x9, x1
100000bfc: b9000fe9    	str	w9, [sp, #0xc]
100000c00: 91006100    	add	x0, x8, #0x18
100000c04: 9400003d    	bl	0x100000cf8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000c08: 14000001    	b	0x100000c0c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000c0c: f94003e0    	ldr	x0, [sp]
100000c10: 940001ea    	bl	0x1000013b8 <___stack_chk_guard+0x1000013b8>
100000c14: 14000001    	b	0x100000c18 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xb4>
100000c18: f9400be0    	ldr	x0, [sp, #0x10]
100000c1c: 940001cf    	bl	0x100001358 <___stack_chk_guard+0x100001358>

0000000100000c20 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000c20: d100c3ff    	sub	sp, sp, #0x30
100000c24: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c28: 910083fd    	add	x29, sp, #0x20
100000c2c: f81f83a0    	stur	x0, [x29, #-0x8]
100000c30: f9000be1    	str	x1, [sp, #0x10]
100000c34: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c38: f90007e0    	str	x0, [sp, #0x8]
100000c3c: f9400be1    	ldr	x1, [sp, #0x10]
100000c40: 94000070    	bl	0x100000e00 <__ZNSt3__114__shared_countC2B8ne200100El>
100000c44: f94007e0    	ldr	x0, [sp, #0x8]
100000c48: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000c4c: f9403d08    	ldr	x8, [x8, #0x78]
100000c50: 91004108    	add	x8, x8, #0x10
100000c54: f9000008    	str	x8, [x0]
100000c58: f9400be8    	ldr	x8, [sp, #0x10]
100000c5c: f9000808    	str	x8, [x0, #0x10]
100000c60: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c64: 9100c3ff    	add	sp, sp, #0x30
100000c68: d65f03c0    	ret

0000000100000c6c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>:
100000c6c: d100c3ff    	sub	sp, sp, #0x30
100000c70: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c74: 910083fd    	add	x29, sp, #0x20
100000c78: f81f83a0    	stur	x0, [x29, #-0x8]
100000c7c: f9000be1    	str	x1, [sp, #0x10]
100000c80: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c84: f90007e0    	str	x0, [sp, #0x8]
100000c88: f9400be1    	ldr	x1, [sp, #0x10]
100000c8c: 94000069    	bl	0x100000e30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>
100000c90: f94007e0    	ldr	x0, [sp, #0x8]
100000c94: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c98: 9100c3ff    	add	sp, sp, #0x30
100000c9c: d65f03c0    	ret

0000000100000ca0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>:
100000ca0: d10083ff    	sub	sp, sp, #0x20
100000ca4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ca8: 910043fd    	add	x29, sp, #0x10
100000cac: f90007e0    	str	x0, [sp, #0x8]
100000cb0: f94007e8    	ldr	x8, [sp, #0x8]
100000cb4: 91006100    	add	x0, x8, #0x18
100000cb8: 9400006a    	bl	0x100000e60 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000cbc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cc0: 910083ff    	add	sp, sp, #0x20
100000cc4: d65f03c0    	ret

0000000100000cc8 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE9constructB8ne200100IiJiEvLi0EEEvRS2_PT_DpOT0_>:
100000cc8: d100c3ff    	sub	sp, sp, #0x30
100000ccc: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000cd0: 910083fd    	add	x29, sp, #0x20
100000cd4: f81f83a0    	stur	x0, [x29, #-0x8]
100000cd8: f9000be1    	str	x1, [sp, #0x10]
100000cdc: f90007e2    	str	x2, [sp, #0x8]
100000ce0: f9400be0    	ldr	x0, [sp, #0x10]
100000ce4: f94007e1    	ldr	x1, [sp, #0x8]
100000ce8: 94000063    	bl	0x100000e74 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000cec: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000cf0: 9100c3ff    	add	sp, sp, #0x30
100000cf4: d65f03c0    	ret

0000000100000cf8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>:
100000cf8: d10083ff    	sub	sp, sp, #0x20
100000cfc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d00: 910043fd    	add	x29, sp, #0x10
100000d04: f90007e0    	str	x0, [sp, #0x8]
100000d08: f94007e0    	ldr	x0, [sp, #0x8]
100000d0c: f90003e0    	str	x0, [sp]
100000d10: 9400006d    	bl	0x100000ec4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>
100000d14: f94003e0    	ldr	x0, [sp]
100000d18: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d1c: 910083ff    	add	sp, sp, #0x20
100000d20: d65f03c0    	ret

0000000100000d24 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000d24: d10083ff    	sub	sp, sp, #0x20
100000d28: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d2c: 910043fd    	add	x29, sp, #0x10
100000d30: f90007e0    	str	x0, [sp, #0x8]
100000d34: f94007e0    	ldr	x0, [sp, #0x8]
100000d38: f90003e0    	str	x0, [sp]
100000d3c: 9400006d    	bl	0x100000ef0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>
100000d40: f94003e0    	ldr	x0, [sp]
100000d44: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d48: 910083ff    	add	sp, sp, #0x20
100000d4c: d65f03c0    	ret

0000000100000d50 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000d50: d10083ff    	sub	sp, sp, #0x20
100000d54: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d58: 910043fd    	add	x29, sp, #0x10
100000d5c: f90007e0    	str	x0, [sp, #0x8]
100000d60: f94007e0    	ldr	x0, [sp, #0x8]
100000d64: f90003e0    	str	x0, [sp]
100000d68: 97ffffef    	bl	0x100000d24 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>
100000d6c: f94003e0    	ldr	x0, [sp]
100000d70: 94000195    	bl	0x1000013c4 <___stack_chk_guard+0x1000013c4>
100000d74: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d78: 910083ff    	add	sp, sp, #0x20
100000d7c: d65f03c0    	ret

0000000100000d80 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000d80: d10083ff    	sub	sp, sp, #0x20
100000d84: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d88: 910043fd    	add	x29, sp, #0x10
100000d8c: f90007e0    	str	x0, [sp, #0x8]
100000d90: f94007e0    	ldr	x0, [sp, #0x8]
100000d94: 9400018f    	bl	0x1000013d0 <___stack_chk_guard+0x1000013d0>
100000d98: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d9c: 910083ff    	add	sp, sp, #0x20
100000da0: d65f03c0    	ret

0000000100000da4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000da4: d100c3ff    	sub	sp, sp, #0x30
100000da8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000dac: 910083fd    	add	x29, sp, #0x20
100000db0: f81f83a0    	stur	x0, [x29, #-0x8]
100000db4: f85f83a0    	ldur	x0, [x29, #-0x8]
100000db8: f90003e0    	str	x0, [sp]
100000dbc: 97ffffb9    	bl	0x100000ca0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000dc0: aa0003e1    	mov	x1, x0
100000dc4: d10027a0    	sub	x0, x29, #0x9
100000dc8: f90007e0    	str	x0, [sp, #0x8]
100000dcc: 97fffed5    	bl	0x100000920 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000dd0: f94003e8    	ldr	x8, [sp]
100000dd4: 91006100    	add	x0, x8, #0x18
100000dd8: 97ffffc8    	bl	0x100000cf8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000ddc: f94003e0    	ldr	x0, [sp]
100000de0: 94000086    	bl	0x100000ff8 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>
100000de4: aa0003e1    	mov	x1, x0
100000de8: f94007e0    	ldr	x0, [sp, #0x8]
100000dec: d2800022    	mov	x2, #0x1                ; =1
100000df0: 94000075    	bl	0x100000fc4 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100000df4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000df8: 9100c3ff    	add	sp, sp, #0x30
100000dfc: d65f03c0    	ret

0000000100000e00 <__ZNSt3__114__shared_countC2B8ne200100El>:
100000e00: d10043ff    	sub	sp, sp, #0x10
100000e04: f90007e0    	str	x0, [sp, #0x8]
100000e08: f90003e1    	str	x1, [sp]
100000e0c: f94007e0    	ldr	x0, [sp, #0x8]
100000e10: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000e14: f9404908    	ldr	x8, [x8, #0x90]
100000e18: 91004108    	add	x8, x8, #0x10
100000e1c: f9000008    	str	x8, [x0]
100000e20: f94003e8    	ldr	x8, [sp]
100000e24: f9000408    	str	x8, [x0, #0x8]
100000e28: 910043ff    	add	sp, sp, #0x10
100000e2c: d65f03c0    	ret

0000000100000e30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>:
100000e30: d100c3ff    	sub	sp, sp, #0x30
100000e34: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e38: 910083fd    	add	x29, sp, #0x20
100000e3c: f81f83a0    	stur	x0, [x29, #-0x8]
100000e40: f9000be1    	str	x1, [sp, #0x10]
100000e44: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e48: f90007e0    	str	x0, [sp, #0x8]
100000e4c: 94000005    	bl	0x100000e60 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000e50: f94007e0    	ldr	x0, [sp, #0x8]
100000e54: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e58: 9100c3ff    	add	sp, sp, #0x30
100000e5c: d65f03c0    	ret

0000000100000e60 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>:
100000e60: d10043ff    	sub	sp, sp, #0x10
100000e64: f90007e0    	str	x0, [sp, #0x8]
100000e68: f94007e0    	ldr	x0, [sp, #0x8]
100000e6c: 910043ff    	add	sp, sp, #0x10
100000e70: d65f03c0    	ret

0000000100000e74 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100000e74: d10083ff    	sub	sp, sp, #0x20
100000e78: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e7c: 910043fd    	add	x29, sp, #0x10
100000e80: f90007e0    	str	x0, [sp, #0x8]
100000e84: f90003e1    	str	x1, [sp]
100000e88: f94007e0    	ldr	x0, [sp, #0x8]
100000e8c: f94003e1    	ldr	x1, [sp]
100000e90: 94000004    	bl	0x100000ea0 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000e94: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e98: 910083ff    	add	sp, sp, #0x20
100000e9c: d65f03c0    	ret

0000000100000ea0 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100000ea0: d10043ff    	sub	sp, sp, #0x10
100000ea4: f90007e0    	str	x0, [sp, #0x8]
100000ea8: f90003e1    	str	x1, [sp]
100000eac: f94007e0    	ldr	x0, [sp, #0x8]
100000eb0: f94003e8    	ldr	x8, [sp]
100000eb4: b9400108    	ldr	w8, [x8]
100000eb8: b9000008    	str	w8, [x0]
100000ebc: 910043ff    	add	sp, sp, #0x10
100000ec0: d65f03c0    	ret

0000000100000ec4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>:
100000ec4: d10083ff    	sub	sp, sp, #0x20
100000ec8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ecc: 910043fd    	add	x29, sp, #0x10
100000ed0: f90007e0    	str	x0, [sp, #0x8]
100000ed4: f94007e0    	ldr	x0, [sp, #0x8]
100000ed8: f90003e0    	str	x0, [sp]
100000edc: 97ffffe1    	bl	0x100000e60 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000ee0: f94003e0    	ldr	x0, [sp]
100000ee4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ee8: 910083ff    	add	sp, sp, #0x20
100000eec: d65f03c0    	ret

0000000100000ef0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>:
100000ef0: d10083ff    	sub	sp, sp, #0x20
100000ef4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ef8: 910043fd    	add	x29, sp, #0x10
100000efc: f90007e0    	str	x0, [sp, #0x8]
100000f00: f94007e8    	ldr	x8, [sp, #0x8]
100000f04: f90003e8    	str	x8, [sp]
100000f08: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000f0c: 91030129    	add	x9, x9, #0xc0
100000f10: 91004129    	add	x9, x9, #0x10
100000f14: f9000109    	str	x9, [x8]
100000f18: 91006100    	add	x0, x8, #0x18
100000f1c: 97ffff77    	bl	0x100000cf8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000f20: f94003e0    	ldr	x0, [sp]
100000f24: 94000125    	bl	0x1000013b8 <___stack_chk_guard+0x1000013b8>
100000f28: f94003e0    	ldr	x0, [sp]
100000f2c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f30: 910083ff    	add	sp, sp, #0x20
100000f34: d65f03c0    	ret

0000000100000f38 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv>:
100000f38: d100c3ff    	sub	sp, sp, #0x30
100000f3c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000f40: 910083fd    	add	x29, sp, #0x20
100000f44: f81f83a0    	stur	x0, [x29, #-0x8]
100000f48: f85f83a0    	ldur	x0, [x29, #-0x8]
100000f4c: f90007e0    	str	x0, [sp, #0x8]
100000f50: 97ffff54    	bl	0x100000ca0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000f54: f94007e0    	ldr	x0, [sp, #0x8]
100000f58: 97fffe49    	bl	0x10000087c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000f5c: aa0003e1    	mov	x1, x0
100000f60: d10027a0    	sub	x0, x29, #0x9
100000f64: 9400011e    	bl	0x1000013dc <___stack_chk_guard+0x1000013dc>
100000f68: 14000001    	b	0x100000f6c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv+0x34>
100000f6c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000f70: 9100c3ff    	add	sp, sp, #0x30
100000f74: d65f03c0    	ret
100000f78: 9400000b    	bl	0x100000fa4 <___clang_call_terminate>

0000000100000f7c <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE7destroyB8ne200100IivLi0EEEvRS2_PT_>:
100000f7c: d10083ff    	sub	sp, sp, #0x20
100000f80: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f84: 910043fd    	add	x29, sp, #0x10
100000f88: f90007e0    	str	x0, [sp, #0x8]
100000f8c: f90003e1    	str	x1, [sp]
100000f90: f94003e0    	ldr	x0, [sp]
100000f94: 94000008    	bl	0x100000fb4 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>
100000f98: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f9c: 910083ff    	add	sp, sp, #0x20
100000fa0: d65f03c0    	ret

0000000100000fa4 <___clang_call_terminate>:
100000fa4: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000fa8: 910003fd    	mov	x29, sp
100000fac: 9400010f    	bl	0x1000013e8 <___stack_chk_guard+0x1000013e8>
100000fb0: 94000111    	bl	0x1000013f4 <___stack_chk_guard+0x1000013f4>

0000000100000fb4 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>:
100000fb4: d10043ff    	sub	sp, sp, #0x10
100000fb8: f90007e0    	str	x0, [sp, #0x8]
100000fbc: 910043ff    	add	sp, sp, #0x10
100000fc0: d65f03c0    	ret

0000000100000fc4 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>:
100000fc4: d100c3ff    	sub	sp, sp, #0x30
100000fc8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000fcc: 910083fd    	add	x29, sp, #0x20
100000fd0: f81f83a0    	stur	x0, [x29, #-0x8]
100000fd4: f9000be1    	str	x1, [sp, #0x10]
100000fd8: f90007e2    	str	x2, [sp, #0x8]
100000fdc: f85f83a0    	ldur	x0, [x29, #-0x8]
100000fe0: f9400be1    	ldr	x1, [sp, #0x10]
100000fe4: f94007e2    	ldr	x2, [sp, #0x8]
100000fe8: 94000009    	bl	0x10000100c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>
100000fec: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ff0: 9100c3ff    	add	sp, sp, #0x30
100000ff4: d65f03c0    	ret

0000000100000ff8 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>:
100000ff8: d10043ff    	sub	sp, sp, #0x10
100000ffc: f90007e0    	str	x0, [sp, #0x8]
100001000: f94007e0    	ldr	x0, [sp, #0x8]
100001004: 910043ff    	add	sp, sp, #0x10
100001008: d65f03c0    	ret

000000010000100c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>:
10000100c: d100c3ff    	sub	sp, sp, #0x30
100001010: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001014: 910083fd    	add	x29, sp, #0x20
100001018: f81f83a0    	stur	x0, [x29, #-0x8]
10000101c: f9000be1    	str	x1, [sp, #0x10]
100001020: f90007e2    	str	x2, [sp, #0x8]
100001024: f9400be0    	ldr	x0, [sp, #0x10]
100001028: f94007e1    	ldr	x1, [sp, #0x8]
10000102c: d2800102    	mov	x2, #0x8                ; =8
100001030: 94000004    	bl	0x100001040 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
100001034: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001038: 9100c3ff    	add	sp, sp, #0x30
10000103c: d65f03c0    	ret

0000000100001040 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
100001040: d10103ff    	sub	sp, sp, #0x40
100001044: a9037bfd    	stp	x29, x30, [sp, #0x30]
100001048: 9100c3fd    	add	x29, sp, #0x30
10000104c: f81f83a0    	stur	x0, [x29, #-0x8]
100001050: f81f03a1    	stur	x1, [x29, #-0x10]
100001054: f9000fe2    	str	x2, [sp, #0x18]
100001058: f85f03a8    	ldur	x8, [x29, #-0x10]
10000105c: d37be908    	lsl	x8, x8, #5
100001060: f9000be8    	str	x8, [sp, #0x10]
100001064: f9400fe0    	ldr	x0, [sp, #0x18]
100001068: 97fffea4    	bl	0x100000af8 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
10000106c: 36000100    	tbz	w0, #0x0, 0x10000108c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x4c>
100001070: 14000001    	b	0x100001074 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x34>
100001074: f9400fe8    	ldr	x8, [sp, #0x18]
100001078: f90007e8    	str	x8, [sp, #0x8]
10000107c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001080: f94007e1    	ldr	x1, [sp, #0x8]
100001084: 94000008    	bl	0x1000010a4 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>
100001088: 14000004    	b	0x100001098 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
10000108c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001090: 94000010    	bl	0x1000010d0 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>
100001094: 14000001    	b	0x100001098 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
100001098: a9437bfd    	ldp	x29, x30, [sp, #0x30]
10000109c: 910103ff    	add	sp, sp, #0x40
1000010a0: d65f03c0    	ret

00000001000010a4 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>:
1000010a4: d10083ff    	sub	sp, sp, #0x20
1000010a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000010ac: 910043fd    	add	x29, sp, #0x10
1000010b0: f90007e0    	str	x0, [sp, #0x8]
1000010b4: f90003e1    	str	x1, [sp]
1000010b8: f94007e0    	ldr	x0, [sp, #0x8]
1000010bc: f94003e1    	ldr	x1, [sp]
1000010c0: 940000d0    	bl	0x100001400 <___stack_chk_guard+0x100001400>
1000010c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000010c8: 910083ff    	add	sp, sp, #0x20
1000010cc: d65f03c0    	ret

00000001000010d0 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>:
1000010d0: d10083ff    	sub	sp, sp, #0x20
1000010d4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000010d8: 910043fd    	add	x29, sp, #0x10
1000010dc: f90007e0    	str	x0, [sp, #0x8]
1000010e0: f94007e0    	ldr	x0, [sp, #0x8]
1000010e4: 940000b8    	bl	0x1000013c4 <___stack_chk_guard+0x1000013c4>
1000010e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000010ec: 910083ff    	add	sp, sp, #0x20
1000010f0: d65f03c0    	ret

00000001000010f4 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>:
1000010f4: d10083ff    	sub	sp, sp, #0x20
1000010f8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000010fc: 910043fd    	add	x29, sp, #0x10
100001100: f90007e0    	str	x0, [sp, #0x8]
100001104: f94007e0    	ldr	x0, [sp, #0x8]
100001108: f90003e0    	str	x0, [sp]
10000110c: 94000009    	bl	0x100001130 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>
100001110: f94003e0    	ldr	x0, [sp]
100001114: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001118: 910083ff    	add	sp, sp, #0x20
10000111c: d65f03c0    	ret

0000000100001120 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>:
100001120: d10043ff    	sub	sp, sp, #0x10
100001124: f90007e0    	str	x0, [sp, #0x8]
100001128: 910043ff    	add	sp, sp, #0x10
10000112c: d65f03c0    	ret

0000000100001130 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>:
100001130: d10043ff    	sub	sp, sp, #0x10
100001134: f90007e0    	str	x0, [sp, #0x8]
100001138: f94007e0    	ldr	x0, [sp, #0x8]
10000113c: f900001f    	str	xzr, [x0]
100001140: f900041f    	str	xzr, [x0, #0x8]
100001144: 910043ff    	add	sp, sp, #0x10
100001148: d65f03c0    	ret

000000010000114c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>:
10000114c: d10043ff    	sub	sp, sp, #0x10
100001150: f90007e0    	str	x0, [sp, #0x8]
100001154: f94007e0    	ldr	x0, [sp, #0x8]
100001158: 910043ff    	add	sp, sp, #0x10
10000115c: d65f03c0    	ret

0000000100001160 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>:
100001160: d10083ff    	sub	sp, sp, #0x20
100001164: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001168: 910043fd    	add	x29, sp, #0x10
10000116c: f90007e0    	str	x0, [sp, #0x8]
100001170: f94007e0    	ldr	x0, [sp, #0x8]
100001174: f90003e0    	str	x0, [sp]
100001178: 94000005    	bl	0x10000118c <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>
10000117c: f94003e0    	ldr	x0, [sp]
100001180: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001184: 910083ff    	add	sp, sp, #0x20
100001188: d65f03c0    	ret

000000010000118c <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>:
10000118c: d10083ff    	sub	sp, sp, #0x20
100001190: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001194: 910043fd    	add	x29, sp, #0x10
100001198: f90007e0    	str	x0, [sp, #0x8]
10000119c: f94007e8    	ldr	x8, [sp, #0x8]
1000011a0: f90003e8    	str	x8, [sp]
1000011a4: f9400908    	ldr	x8, [x8, #0x10]
1000011a8: b40000e8    	cbz	x8, 0x1000011c4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
1000011ac: 14000001    	b	0x1000011b0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x24>
1000011b0: f94003e0    	ldr	x0, [sp]
1000011b4: f9400801    	ldr	x1, [x0, #0x10]
1000011b8: f9400402    	ldr	x2, [x0, #0x8]
1000011bc: 97ffff82    	bl	0x100000fc4 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
1000011c0: 14000001    	b	0x1000011c4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
1000011c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000011c8: 910083ff    	add	sp, sp, #0x20
1000011cc: d65f03c0    	ret

00000001000011d0 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>:
1000011d0: d10083ff    	sub	sp, sp, #0x20
1000011d4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000011d8: 910043fd    	add	x29, sp, #0x10
1000011dc: f90007e0    	str	x0, [sp, #0x8]
1000011e0: f94007e0    	ldr	x0, [sp, #0x8]
1000011e4: f90003e0    	str	x0, [sp]
1000011e8: 94000005    	bl	0x1000011fc <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
1000011ec: f94003e0    	ldr	x0, [sp]
1000011f0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000011f4: 910083ff    	add	sp, sp, #0x20
1000011f8: d65f03c0    	ret

00000001000011fc <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
1000011fc: d10043ff    	sub	sp, sp, #0x10
100001200: f90007e0    	str	x0, [sp, #0x8]
100001204: f94007e0    	ldr	x0, [sp, #0x8]
100001208: 910043ff    	add	sp, sp, #0x10
10000120c: d65f03c0    	ret

0000000100001210 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>:
100001210: d100c3ff    	sub	sp, sp, #0x30
100001214: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001218: 910083fd    	add	x29, sp, #0x20
10000121c: f9000be0    	str	x0, [sp, #0x10]
100001220: f9400be8    	ldr	x8, [sp, #0x10]
100001224: f90007e8    	str	x8, [sp, #0x8]
100001228: aa0803e9    	mov	x9, x8
10000122c: f81f83a9    	stur	x9, [x29, #-0x8]
100001230: f9400508    	ldr	x8, [x8, #0x8]
100001234: b40000c8    	cbz	x8, 0x10000124c <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
100001238: 14000001    	b	0x10000123c <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x2c>
10000123c: f94007e8    	ldr	x8, [sp, #0x8]
100001240: f9400500    	ldr	x0, [x8, #0x8]
100001244: 94000006    	bl	0x10000125c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
100001248: 14000001    	b	0x10000124c <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
10000124c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001250: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001254: 9100c3ff    	add	sp, sp, #0x30
100001258: d65f03c0    	ret

000000010000125c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
10000125c: d10083ff    	sub	sp, sp, #0x20
100001260: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001264: 910043fd    	add	x29, sp, #0x10
100001268: f90007e0    	str	x0, [sp, #0x8]
10000126c: f94007e0    	ldr	x0, [sp, #0x8]
100001270: f90003e0    	str	x0, [sp]
100001274: 94000009    	bl	0x100001298 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
100001278: 360000a0    	tbz	w0, #0x0, 0x10000128c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
10000127c: 14000001    	b	0x100001280 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
100001280: f94003e0    	ldr	x0, [sp]
100001284: 94000062    	bl	0x10000140c <___stack_chk_guard+0x10000140c>
100001288: 14000001    	b	0x10000128c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
10000128c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001290: 910083ff    	add	sp, sp, #0x20
100001294: d65f03c0    	ret

0000000100001298 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
100001298: d100c3ff    	sub	sp, sp, #0x30
10000129c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000012a0: 910083fd    	add	x29, sp, #0x20
1000012a4: f9000be0    	str	x0, [sp, #0x10]
1000012a8: f9400be8    	ldr	x8, [sp, #0x10]
1000012ac: f90007e8    	str	x8, [sp, #0x8]
1000012b0: 91002100    	add	x0, x8, #0x8
1000012b4: 94000017    	bl	0x100001310 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
1000012b8: b1000408    	adds	x8, x0, #0x1
1000012bc: 54000161    	b.ne	0x1000012e8 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
1000012c0: 14000001    	b	0x1000012c4 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
1000012c4: f94007e0    	ldr	x0, [sp, #0x8]
1000012c8: f9400008    	ldr	x8, [x0]
1000012cc: f9400908    	ldr	x8, [x8, #0x10]
1000012d0: d63f0100    	blr	x8
1000012d4: 52800028    	mov	w8, #0x1                ; =1
1000012d8: 12000108    	and	w8, w8, #0x1
1000012dc: 12000108    	and	w8, w8, #0x1
1000012e0: 381ff3a8    	sturb	w8, [x29, #-0x1]
1000012e4: 14000006    	b	0x1000012fc <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
1000012e8: 52800008    	mov	w8, #0x0                ; =0
1000012ec: 12000108    	and	w8, w8, #0x1
1000012f0: 12000108    	and	w8, w8, #0x1
1000012f4: 381ff3a8    	sturb	w8, [x29, #-0x1]
1000012f8: 14000001    	b	0x1000012fc <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
1000012fc: 385ff3a8    	ldurb	w8, [x29, #-0x1]
100001300: 12000100    	and	w0, w8, #0x1
100001304: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001308: 9100c3ff    	add	sp, sp, #0x30
10000130c: d65f03c0    	ret

0000000100001310 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
100001310: d10083ff    	sub	sp, sp, #0x20
100001314: f9000fe0    	str	x0, [sp, #0x18]
100001318: f9400fe8    	ldr	x8, [sp, #0x18]
10000131c: 92800009    	mov	x9, #-0x1               ; =-1
100001320: f9000be9    	str	x9, [sp, #0x10]
100001324: f9400be9    	ldr	x9, [sp, #0x10]
100001328: f8e90108    	ldaddal	x9, x8, [x8]
10000132c: 8b090108    	add	x8, x8, x9
100001330: f90007e8    	str	x8, [sp, #0x8]
100001334: f94007e0    	ldr	x0, [sp, #0x8]
100001338: 910083ff    	add	sp, sp, #0x20
10000133c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001340 <__stubs>:
100001340: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001344: f9400a10    	ldr	x16, [x16, #0x10]
100001348: d61f0200    	br	x16
10000134c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001350: f9400e10    	ldr	x16, [x16, #0x18]
100001354: d61f0200    	br	x16
100001358: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000135c: f9401210    	ldr	x16, [x16, #0x20]
100001360: d61f0200    	br	x16
100001364: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001368: f9401610    	ldr	x16, [x16, #0x28]
10000136c: d61f0200    	br	x16
100001370: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001374: f9401a10    	ldr	x16, [x16, #0x30]
100001378: d61f0200    	br	x16
10000137c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001380: f9401e10    	ldr	x16, [x16, #0x38]
100001384: d61f0200    	br	x16
100001388: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000138c: f9402a10    	ldr	x16, [x16, #0x50]
100001390: d61f0200    	br	x16
100001394: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001398: f9402e10    	ldr	x16, [x16, #0x58]
10000139c: d61f0200    	br	x16
1000013a0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013a4: f9403210    	ldr	x16, [x16, #0x60]
1000013a8: d61f0200    	br	x16
1000013ac: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013b0: f9403610    	ldr	x16, [x16, #0x68]
1000013b4: d61f0200    	br	x16
1000013b8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013bc: f9403a10    	ldr	x16, [x16, #0x70]
1000013c0: d61f0200    	br	x16
1000013c4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013c8: f9404210    	ldr	x16, [x16, #0x80]
1000013cc: d61f0200    	br	x16
1000013d0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013d4: f9404610    	ldr	x16, [x16, #0x88]
1000013d8: d61f0200    	br	x16
1000013dc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013e0: f9404e10    	ldr	x16, [x16, #0x98]
1000013e4: d61f0200    	br	x16
1000013e8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013ec: f9405210    	ldr	x16, [x16, #0xa0]
1000013f0: d61f0200    	br	x16
1000013f4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000013f8: f9405610    	ldr	x16, [x16, #0xa8]
1000013fc: d61f0200    	br	x16
100001400: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001404: f9405a10    	ldr	x16, [x16, #0xb0]
100001408: d61f0200    	br	x16
10000140c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001410: f9405e10    	ldr	x16, [x16, #0xb8]
100001414: d61f0200    	br	x16
