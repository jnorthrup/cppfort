
/Users/jim/work/cppfort/micro-tests/results/memory/mem066-weak-ptr-expired/mem066-weak-ptr-expired_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z21test_weak_ptr_expiredv>:
100000538: d10183ff    	sub	sp, sp, #0x60
10000053c: a9057bfd    	stp	x29, x30, [sp, #0x50]
100000540: 910143fd    	add	x29, sp, #0x50
100000544: d10043a0    	sub	x0, x29, #0x10
100000548: 94000023    	bl	0x1000005d4 <__ZNSt3__18weak_ptrIiEC1Ev>
10000054c: d10093a0    	sub	x0, x29, #0x24
100000550: 52800548    	mov	w8, #0x2a               ; =42
100000554: b81dc3a8    	stur	w8, [x29, #-0x24]
100000558: d10083a8    	sub	x8, x29, #0x20
10000055c: 94000029    	bl	0x100000600 <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>
100000560: 14000001    	b	0x100000564 <__Z21test_weak_ptr_expiredv+0x2c>
100000564: d10043a0    	sub	x0, x29, #0x10
100000568: f9000be0    	str	x0, [sp, #0x10]
10000056c: d10083a1    	sub	x1, x29, #0x20
100000570: f90007e1    	str	x1, [sp, #0x8]
100000574: 94000467    	bl	0x100001710 <___stack_chk_guard+0x100001710>
100000578: f94007e0    	ldr	x0, [sp, #0x8]
10000057c: 94000045    	bl	0x100000690 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000580: f9400be0    	ldr	x0, [sp, #0x10]
100000584: 9400004e    	bl	0x1000006bc <__ZNKSt3__18weak_ptrIiE7expiredB8ne200100Ev>
100000588: aa0003e9    	mov	x9, x0
10000058c: f9400be0    	ldr	x0, [sp, #0x10]
100000590: 52800008    	mov	w8, #0x0                ; =0
100000594: 72000129    	ands	w9, w9, #0x1
100000598: 1a9f0508    	csinc	w8, w8, wzr, eq
10000059c: b9001be8    	str	w8, [sp, #0x18]
1000005a0: 9400005e    	bl	0x100000718 <__ZNSt3__18weak_ptrIiED1Ev>
1000005a4: b9401be0    	ldr	w0, [sp, #0x18]
1000005a8: a9457bfd    	ldp	x29, x30, [sp, #0x50]
1000005ac: 910183ff    	add	sp, sp, #0x60
1000005b0: d65f03c0    	ret
1000005b4: f90013e0    	str	x0, [sp, #0x20]
1000005b8: aa0103e8    	mov	x8, x1
1000005bc: b9001fe8    	str	w8, [sp, #0x1c]
1000005c0: d10043a0    	sub	x0, x29, #0x10
1000005c4: 94000055    	bl	0x100000718 <__ZNSt3__18weak_ptrIiED1Ev>
1000005c8: 14000001    	b	0x1000005cc <__Z21test_weak_ptr_expiredv+0x94>
1000005cc: f94013e0    	ldr	x0, [sp, #0x20]
1000005d0: 94000453    	bl	0x10000171c <___stack_chk_guard+0x10000171c>

00000001000005d4 <__ZNSt3__18weak_ptrIiEC1Ev>:
1000005d4: d10083ff    	sub	sp, sp, #0x20
1000005d8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000005dc: 910043fd    	add	x29, sp, #0x10
1000005e0: f90007e0    	str	x0, [sp, #0x8]
1000005e4: f94007e0    	ldr	x0, [sp, #0x8]
1000005e8: f90003e0    	str	x0, [sp]
1000005ec: 9400005e    	bl	0x100000764 <__ZNSt3__18weak_ptrIiEC2Ev>
1000005f0: f94003e0    	ldr	x0, [sp]
1000005f4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005f8: 910083ff    	add	sp, sp, #0x20
1000005fc: d65f03c0    	ret

0000000100000600 <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>:
100000600: d10103ff    	sub	sp, sp, #0x40
100000604: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000608: 9100c3fd    	add	x29, sp, #0x30
10000060c: f9000be8    	str	x8, [sp, #0x10]
100000610: f81f83a8    	stur	x8, [x29, #-0x8]
100000614: f81f03a0    	stur	x0, [x29, #-0x10]
100000618: d10047a0    	sub	x0, x29, #0x11
10000061c: f90007e0    	str	x0, [sp, #0x8]
100000620: 940000a2    	bl	0x1000008a8 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>
100000624: f94007e0    	ldr	x0, [sp, #0x8]
100000628: f9400be8    	ldr	x8, [sp, #0x10]
10000062c: f85f03a1    	ldur	x1, [x29, #-0x10]
100000630: 94000067    	bl	0x1000007cc <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>
100000634: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000638: 910103ff    	add	sp, sp, #0x40
10000063c: d65f03c0    	ret

0000000100000640 <__ZNSt3__18weak_ptrIiEaSIiLi0EEERS1_RKNS_10shared_ptrIT_EE>:
100000640: d10103ff    	sub	sp, sp, #0x40
100000644: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000648: 9100c3fd    	add	x29, sp, #0x30
10000064c: f81f83a0    	stur	x0, [x29, #-0x8]
100000650: f81f03a1    	stur	x1, [x29, #-0x10]
100000654: f85f83a8    	ldur	x8, [x29, #-0x8]
100000658: f90007e8    	str	x8, [sp, #0x8]
10000065c: f85f03a1    	ldur	x1, [x29, #-0x10]
100000660: 910043e0    	add	x0, sp, #0x10
100000664: f90003e0    	str	x0, [sp]
100000668: 9400039e    	bl	0x1000014e0 <__ZNSt3__18weak_ptrIiEC1IiLi0EEERKNS_10shared_ptrIT_EE>
10000066c: f94003e0    	ldr	x0, [sp]
100000670: f94007e1    	ldr	x1, [sp, #0x8]
100000674: 940003a8    	bl	0x100001514 <__ZNSt3__18weak_ptrIiE4swapERS1_>
100000678: f94003e0    	ldr	x0, [sp]
10000067c: 94000027    	bl	0x100000718 <__ZNSt3__18weak_ptrIiED1Ev>
100000680: f94007e0    	ldr	x0, [sp, #0x8]
100000684: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000688: 910103ff    	add	sp, sp, #0x40
10000068c: d65f03c0    	ret

0000000100000690 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>:
100000690: d10083ff    	sub	sp, sp, #0x20
100000694: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000698: 910043fd    	add	x29, sp, #0x10
10000069c: f90007e0    	str	x0, [sp, #0x8]
1000006a0: f94007e0    	ldr	x0, [sp, #0x8]
1000006a4: f90003e0    	str	x0, [sp]
1000006a8: 94000342    	bl	0x1000013b0 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>
1000006ac: f94003e0    	ldr	x0, [sp]
1000006b0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000006b4: 910083ff    	add	sp, sp, #0x20
1000006b8: d65f03c0    	ret

00000001000006bc <__ZNKSt3__18weak_ptrIiE7expiredB8ne200100Ev>:
1000006bc: d100c3ff    	sub	sp, sp, #0x30
1000006c0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000006c4: 910083fd    	add	x29, sp, #0x20
1000006c8: f81f83a0    	stur	x0, [x29, #-0x8]
1000006cc: f85f83a8    	ldur	x8, [x29, #-0x8]
1000006d0: f90007e8    	str	x8, [sp, #0x8]
1000006d4: f9400508    	ldr	x8, [x8, #0x8]
1000006d8: 52800029    	mov	w9, #0x1                ; =1
1000006dc: b81f43a9    	stur	w9, [x29, #-0xc]
1000006e0: b4000128    	cbz	x8, 0x100000704 <__ZNKSt3__18weak_ptrIiE7expiredB8ne200100Ev+0x48>
1000006e4: 14000001    	b	0x1000006e8 <__ZNKSt3__18weak_ptrIiE7expiredB8ne200100Ev+0x2c>
1000006e8: f94007e8    	ldr	x8, [sp, #0x8]
1000006ec: f9400500    	ldr	x0, [x8, #0x8]
1000006f0: 940003e8    	bl	0x100001690 <__ZNKSt3__119__shared_weak_count9use_countB8ne200100Ev>
1000006f4: f1000008    	subs	x8, x0, #0x0
1000006f8: 1a9f17e8    	cset	w8, eq
1000006fc: b81f43a8    	stur	w8, [x29, #-0xc]
100000700: 14000001    	b	0x100000704 <__ZNKSt3__18weak_ptrIiE7expiredB8ne200100Ev+0x48>
100000704: b85f43a8    	ldur	w8, [x29, #-0xc]
100000708: 12000100    	and	w0, w8, #0x1
10000070c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000710: 9100c3ff    	add	sp, sp, #0x30
100000714: d65f03c0    	ret

0000000100000718 <__ZNSt3__18weak_ptrIiED1Ev>:
100000718: d10083ff    	sub	sp, sp, #0x20
10000071c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000720: 910043fd    	add	x29, sp, #0x10
100000724: f90007e0    	str	x0, [sp, #0x8]
100000728: f94007e0    	ldr	x0, [sp, #0x8]
10000072c: f90003e0    	str	x0, [sp]
100000730: 94000014    	bl	0x100000780 <__ZNSt3__18weak_ptrIiED2Ev>
100000734: f94003e0    	ldr	x0, [sp]
100000738: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000073c: 910083ff    	add	sp, sp, #0x20
100000740: d65f03c0    	ret

0000000100000744 <_main>:
100000744: d10083ff    	sub	sp, sp, #0x20
100000748: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000074c: 910043fd    	add	x29, sp, #0x10
100000750: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000754: 97ffff79    	bl	0x100000538 <__Z21test_weak_ptr_expiredv>
100000758: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000075c: 910083ff    	add	sp, sp, #0x20
100000760: d65f03c0    	ret

0000000100000764 <__ZNSt3__18weak_ptrIiEC2Ev>:
100000764: d10043ff    	sub	sp, sp, #0x10
100000768: f90007e0    	str	x0, [sp, #0x8]
10000076c: f94007e0    	ldr	x0, [sp, #0x8]
100000770: f900001f    	str	xzr, [x0]
100000774: f900041f    	str	xzr, [x0, #0x8]
100000778: 910043ff    	add	sp, sp, #0x10
10000077c: d65f03c0    	ret

0000000100000780 <__ZNSt3__18weak_ptrIiED2Ev>:
100000780: d100c3ff    	sub	sp, sp, #0x30
100000784: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000788: 910083fd    	add	x29, sp, #0x20
10000078c: f9000be0    	str	x0, [sp, #0x10]
100000790: f9400be8    	ldr	x8, [sp, #0x10]
100000794: f90007e8    	str	x8, [sp, #0x8]
100000798: aa0803e9    	mov	x9, x8
10000079c: f81f83a9    	stur	x9, [x29, #-0x8]
1000007a0: f9400508    	ldr	x8, [x8, #0x8]
1000007a4: b40000c8    	cbz	x8, 0x1000007bc <__ZNSt3__18weak_ptrIiED2Ev+0x3c>
1000007a8: 14000001    	b	0x1000007ac <__ZNSt3__18weak_ptrIiED2Ev+0x2c>
1000007ac: f94007e8    	ldr	x8, [sp, #0x8]
1000007b0: f9400500    	ldr	x0, [x8, #0x8]
1000007b4: 940003dd    	bl	0x100001728 <___stack_chk_guard+0x100001728>
1000007b8: 14000001    	b	0x1000007bc <__ZNSt3__18weak_ptrIiED2Ev+0x3c>
1000007bc: f85f83a0    	ldur	x0, [x29, #-0x8]
1000007c0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000007c4: 9100c3ff    	add	sp, sp, #0x30
1000007c8: d65f03c0    	ret

00000001000007cc <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>:
1000007cc: d10203ff    	sub	sp, sp, #0x80
1000007d0: a9077bfd    	stp	x29, x30, [sp, #0x70]
1000007d4: 9101c3fd    	add	x29, sp, #0x70
1000007d8: f9000be8    	str	x8, [sp, #0x10]
1000007dc: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000007e0: f9401129    	ldr	x9, [x9, #0x20]
1000007e4: f9400129    	ldr	x9, [x9]
1000007e8: f81f83a9    	stur	x9, [x29, #-0x8]
1000007ec: f81d83a8    	stur	x8, [x29, #-0x28]
1000007f0: f81d03a0    	stur	x0, [x29, #-0x30]
1000007f4: f9001fe1    	str	x1, [sp, #0x38]
1000007f8: d10083a0    	sub	x0, x29, #0x20
1000007fc: d2800021    	mov	x1, #0x1                ; =1
100000800: 94000035    	bl	0x1000008d4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>
100000804: 14000001    	b	0x100000808 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x3c>
100000808: d10083a0    	sub	x0, x29, #0x20
10000080c: 9400003f    	bl	0x100000908 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>
100000810: f9401fe1    	ldr	x1, [sp, #0x38]
100000814: 94000043    	bl	0x100000920 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>
100000818: 14000001    	b	0x10000081c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x50>
10000081c: d10083a0    	sub	x0, x29, #0x20
100000820: f90007e0    	str	x0, [sp, #0x8]
100000824: 9400004c    	bl	0x100000954 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>
100000828: f9000fe0    	str	x0, [sp, #0x18]
10000082c: f9400fe0    	ldr	x0, [sp, #0x18]
100000830: 9400007b    	bl	0x100000a1c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000834: f9400be8    	ldr	x8, [sp, #0x10]
100000838: f9400fe1    	ldr	x1, [sp, #0x18]
10000083c: 940003be    	bl	0x100001734 <___stack_chk_guard+0x100001734>
100000840: f94007e0    	ldr	x0, [sp, #0x8]
100000844: 94000080    	bl	0x100000a44 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
100000848: f85f83a9    	ldur	x9, [x29, #-0x8]
10000084c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000850: f9401108    	ldr	x8, [x8, #0x20]
100000854: f9400108    	ldr	x8, [x8]
100000858: eb090108    	subs	x8, x8, x9
10000085c: 54000060    	b.eq	0x100000868 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x9c>
100000860: 14000001    	b	0x100000864 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x98>
100000864: 940003b7    	bl	0x100001740 <___stack_chk_guard+0x100001740>
100000868: a9477bfd    	ldp	x29, x30, [sp, #0x70]
10000086c: 910203ff    	add	sp, sp, #0x80
100000870: d65f03c0    	ret
100000874: f90017e0    	str	x0, [sp, #0x28]
100000878: aa0103e8    	mov	x8, x1
10000087c: b90027e8    	str	w8, [sp, #0x24]
100000880: d10083a0    	sub	x0, x29, #0x20
100000884: 94000070    	bl	0x100000a44 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
100000888: 14000001    	b	0x10000088c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xc0>
10000088c: f94017e0    	ldr	x0, [sp, #0x28]
100000890: f90003e0    	str	x0, [sp]
100000894: 14000003    	b	0x1000008a0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
100000898: f90003e0    	str	x0, [sp]
10000089c: 14000001    	b	0x1000008a0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
1000008a0: f94003e0    	ldr	x0, [sp]
1000008a4: 9400039e    	bl	0x10000171c <___stack_chk_guard+0x10000171c>

00000001000008a8 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>:
1000008a8: d10083ff    	sub	sp, sp, #0x20
1000008ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000008b0: 910043fd    	add	x29, sp, #0x10
1000008b4: f90007e0    	str	x0, [sp, #0x8]
1000008b8: f94007e0    	ldr	x0, [sp, #0x8]
1000008bc: f90003e0    	str	x0, [sp]
1000008c0: 940002ac    	bl	0x100001370 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>
1000008c4: f94003e0    	ldr	x0, [sp]
1000008c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000008cc: 910083ff    	add	sp, sp, #0x20
1000008d0: d65f03c0    	ret

00000001000008d4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>:
1000008d4: d100c3ff    	sub	sp, sp, #0x30
1000008d8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000008dc: 910083fd    	add	x29, sp, #0x20
1000008e0: f9000be0    	str	x0, [sp, #0x10]
1000008e4: f90007e1    	str	x1, [sp, #0x8]
1000008e8: f9400be0    	ldr	x0, [sp, #0x10]
1000008ec: f90003e0    	str	x0, [sp]
1000008f0: f94007e1    	ldr	x1, [sp, #0x8]
1000008f4: 9400005f    	bl	0x100000a70 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>
1000008f8: f94003e0    	ldr	x0, [sp]
1000008fc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000900: 9100c3ff    	add	sp, sp, #0x30
100000904: d65f03c0    	ret

0000000100000908 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>:
100000908: d10043ff    	sub	sp, sp, #0x10
10000090c: f90007e0    	str	x0, [sp, #0x8]
100000910: f94007e8    	ldr	x8, [sp, #0x8]
100000914: f9400900    	ldr	x0, [x8, #0x10]
100000918: 910043ff    	add	sp, sp, #0x10
10000091c: d65f03c0    	ret

0000000100000920 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000920: d100c3ff    	sub	sp, sp, #0x30
100000924: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000928: 910083fd    	add	x29, sp, #0x20
10000092c: f9000be0    	str	x0, [sp, #0x10]
100000930: f90007e1    	str	x1, [sp, #0x8]
100000934: f9400be0    	ldr	x0, [sp, #0x10]
100000938: f90003e0    	str	x0, [sp]
10000093c: f94007e1    	ldr	x1, [sp, #0x8]
100000940: 940000f1    	bl	0x100000d04 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>
100000944: f94003e0    	ldr	x0, [sp]
100000948: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000094c: 9100c3ff    	add	sp, sp, #0x30
100000950: d65f03c0    	ret

0000000100000954 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>:
100000954: d10043ff    	sub	sp, sp, #0x10
100000958: f90007e0    	str	x0, [sp, #0x8]
10000095c: f94007e8    	ldr	x8, [sp, #0x8]
100000960: f9400909    	ldr	x9, [x8, #0x10]
100000964: f90003e9    	str	x9, [sp]
100000968: f900091f    	str	xzr, [x8, #0x10]
10000096c: f94003e0    	ldr	x0, [sp]
100000970: 910043ff    	add	sp, sp, #0x10
100000974: d65f03c0    	ret

0000000100000978 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_>:
100000978: d10143ff    	sub	sp, sp, #0x50
10000097c: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000980: 910103fd    	add	x29, sp, #0x40
100000984: f9000fe8    	str	x8, [sp, #0x18]
100000988: aa0003e8    	mov	x8, x0
10000098c: f9400fe0    	ldr	x0, [sp, #0x18]
100000990: aa0003e9    	mov	x9, x0
100000994: f81f83a9    	stur	x9, [x29, #-0x8]
100000998: f81f03a8    	stur	x8, [x29, #-0x10]
10000099c: f81e83a1    	stur	x1, [x29, #-0x18]
1000009a0: 52800008    	mov	w8, #0x0                ; =0
1000009a4: 52800029    	mov	w9, #0x1                ; =1
1000009a8: b90023e9    	str	w9, [sp, #0x20]
1000009ac: 12000108    	and	w8, w8, #0x1
1000009b0: 12000108    	and	w8, w8, #0x1
1000009b4: 381e73a8    	sturb	w8, [x29, #-0x19]
1000009b8: 94000237    	bl	0x100001294 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>
1000009bc: f9400fe0    	ldr	x0, [sp, #0x18]
1000009c0: f85f03a8    	ldur	x8, [x29, #-0x10]
1000009c4: f9000008    	str	x8, [x0]
1000009c8: f85e83a8    	ldur	x8, [x29, #-0x18]
1000009cc: f9000408    	str	x8, [x0, #0x8]
1000009d0: f940000a    	ldr	x10, [x0]
1000009d4: f9400008    	ldr	x8, [x0]
1000009d8: 910003e9    	mov	x9, sp
1000009dc: f900012a    	str	x10, [x9]
1000009e0: f9000528    	str	x8, [x9, #0x8]
1000009e4: 94000237    	bl	0x1000012c0 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>
1000009e8: b94023e9    	ldr	w9, [sp, #0x20]
1000009ec: 12000128    	and	w8, w9, #0x1
1000009f0: 0a090108    	and	w8, w8, w9
1000009f4: 381e73a8    	sturb	w8, [x29, #-0x19]
1000009f8: 385e73a8    	ldurb	w8, [x29, #-0x19]
1000009fc: 370000a8    	tbnz	w8, #0x0, 0x100000a10 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
100000a00: 14000001    	b	0x100000a04 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x8c>
100000a04: f9400fe0    	ldr	x0, [sp, #0x18]
100000a08: 97ffff22    	bl	0x100000690 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000a0c: 14000001    	b	0x100000a10 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
100000a10: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000a14: 910143ff    	add	sp, sp, #0x50
100000a18: d65f03c0    	ret

0000000100000a1c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>:
100000a1c: d10083ff    	sub	sp, sp, #0x20
100000a20: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a24: 910043fd    	add	x29, sp, #0x10
100000a28: f90007e0    	str	x0, [sp, #0x8]
100000a2c: f94007e8    	ldr	x8, [sp, #0x8]
100000a30: 91006100    	add	x0, x8, #0x18
100000a34: 9400022e    	bl	0x1000012ec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>
100000a38: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a3c: 910083ff    	add	sp, sp, #0x20
100000a40: d65f03c0    	ret

0000000100000a44 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>:
100000a44: d10083ff    	sub	sp, sp, #0x20
100000a48: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a4c: 910043fd    	add	x29, sp, #0x10
100000a50: f90007e0    	str	x0, [sp, #0x8]
100000a54: f94007e0    	ldr	x0, [sp, #0x8]
100000a58: f90003e0    	str	x0, [sp]
100000a5c: 94000229    	bl	0x100001300 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>
100000a60: f94003e0    	ldr	x0, [sp]
100000a64: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a68: 910083ff    	add	sp, sp, #0x20
100000a6c: d65f03c0    	ret

0000000100000a70 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>:
100000a70: d100c3ff    	sub	sp, sp, #0x30
100000a74: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a78: 910083fd    	add	x29, sp, #0x20
100000a7c: f9000be0    	str	x0, [sp, #0x10]
100000a80: f90007e1    	str	x1, [sp, #0x8]
100000a84: f9400be0    	ldr	x0, [sp, #0x10]
100000a88: f90003e0    	str	x0, [sp]
100000a8c: d10007a1    	sub	x1, x29, #0x1
100000a90: 9400000c    	bl	0x100000ac0 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000a94: f94003e0    	ldr	x0, [sp]
100000a98: f94007e8    	ldr	x8, [sp, #0x8]
100000a9c: f9000408    	str	x8, [x0, #0x8]
100000aa0: f9400401    	ldr	x1, [x0, #0x8]
100000aa4: 94000014    	bl	0x100000af4 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>
100000aa8: aa0003e8    	mov	x8, x0
100000aac: f94003e0    	ldr	x0, [sp]
100000ab0: f9000808    	str	x8, [x0, #0x10]
100000ab4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ab8: 9100c3ff    	add	sp, sp, #0x30
100000abc: d65f03c0    	ret

0000000100000ac0 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>:
100000ac0: d100c3ff    	sub	sp, sp, #0x30
100000ac4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000ac8: 910083fd    	add	x29, sp, #0x20
100000acc: f81f83a0    	stur	x0, [x29, #-0x8]
100000ad0: f9000be1    	str	x1, [sp, #0x10]
100000ad4: f85f83a0    	ldur	x0, [x29, #-0x8]
100000ad8: f90007e0    	str	x0, [sp, #0x8]
100000adc: f9400be1    	ldr	x1, [sp, #0x10]
100000ae0: 94000010    	bl	0x100000b20 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>
100000ae4: f94007e0    	ldr	x0, [sp, #0x8]
100000ae8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000aec: 9100c3ff    	add	sp, sp, #0x30
100000af0: d65f03c0    	ret

0000000100000af4 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>:
100000af4: d10083ff    	sub	sp, sp, #0x20
100000af8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000afc: 910043fd    	add	x29, sp, #0x10
100000b00: f90007e0    	str	x0, [sp, #0x8]
100000b04: f90003e1    	str	x1, [sp]
100000b08: f94007e0    	ldr	x0, [sp, #0x8]
100000b0c: f94003e1    	ldr	x1, [sp]
100000b10: 94000015    	bl	0x100000b64 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>
100000b14: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000b18: 910083ff    	add	sp, sp, #0x20
100000b1c: d65f03c0    	ret

0000000100000b20 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>:
100000b20: d100c3ff    	sub	sp, sp, #0x30
100000b24: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000b28: 910083fd    	add	x29, sp, #0x20
100000b2c: f81f83a0    	stur	x0, [x29, #-0x8]
100000b30: f9000be1    	str	x1, [sp, #0x10]
100000b34: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b38: f90007e0    	str	x0, [sp, #0x8]
100000b3c: 94000005    	bl	0x100000b50 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>
100000b40: f94007e0    	ldr	x0, [sp, #0x8]
100000b44: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000b48: 9100c3ff    	add	sp, sp, #0x30
100000b4c: d65f03c0    	ret

0000000100000b50 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>:
100000b50: d10043ff    	sub	sp, sp, #0x10
100000b54: f90007e0    	str	x0, [sp, #0x8]
100000b58: f94007e0    	ldr	x0, [sp, #0x8]
100000b5c: 910043ff    	add	sp, sp, #0x10
100000b60: d65f03c0    	ret

0000000100000b64 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>:
100000b64: d100c3ff    	sub	sp, sp, #0x30
100000b68: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000b6c: 910083fd    	add	x29, sp, #0x20
100000b70: f81f83a0    	stur	x0, [x29, #-0x8]
100000b74: f9000be1    	str	x1, [sp, #0x10]
100000b78: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b7c: f9400be8    	ldr	x8, [sp, #0x10]
100000b80: f90007e8    	str	x8, [sp, #0x8]
100000b84: 940002f2    	bl	0x10000174c <___stack_chk_guard+0x10000174c>
100000b88: f94007e8    	ldr	x8, [sp, #0x8]
100000b8c: eb000108    	subs	x8, x8, x0
100000b90: 54000069    	b.ls	0x100000b9c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x38>
100000b94: 14000001    	b	0x100000b98 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x34>
100000b98: 94000011    	bl	0x100000bdc <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
100000b9c: f9400be0    	ldr	x0, [sp, #0x10]
100000ba0: d2800101    	mov	x1, #0x8                ; =8
100000ba4: 9400001b    	bl	0x100000c10 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>
100000ba8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000bac: 9100c3ff    	add	sp, sp, #0x30
100000bb0: d65f03c0    	ret

0000000100000bb4 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8max_sizeB8ne200100IS5_vLi0EEEmRKS5_>:
100000bb4: d10083ff    	sub	sp, sp, #0x20
100000bb8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bbc: 910043fd    	add	x29, sp, #0x10
100000bc0: f90007e0    	str	x0, [sp, #0x8]
100000bc4: 9400002e    	bl	0x100000c7c <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000bc8: d2800408    	mov	x8, #0x20               ; =32
100000bcc: 9ac80800    	udiv	x0, x0, x8
100000bd0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000bd4: 910083ff    	add	sp, sp, #0x20
100000bd8: d65f03c0    	ret

0000000100000bdc <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000bdc: d10083ff    	sub	sp, sp, #0x20
100000be0: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000be4: 910043fd    	add	x29, sp, #0x10
100000be8: d2800100    	mov	x0, #0x8                ; =8
100000bec: 940002db    	bl	0x100001758 <___stack_chk_guard+0x100001758>
100000bf0: f90007e0    	str	x0, [sp, #0x8]
100000bf4: 940002dc    	bl	0x100001764 <___stack_chk_guard+0x100001764>
100000bf8: f94007e0    	ldr	x0, [sp, #0x8]
100000bfc: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000c00: f9402821    	ldr	x1, [x1, #0x50]
100000c04: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000c08: f9402c42    	ldr	x2, [x2, #0x58]
100000c0c: 940002d9    	bl	0x100001770 <___stack_chk_guard+0x100001770>

0000000100000c10 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>:
100000c10: d10103ff    	sub	sp, sp, #0x40
100000c14: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000c18: 9100c3fd    	add	x29, sp, #0x30
100000c1c: f81f03a0    	stur	x0, [x29, #-0x10]
100000c20: f9000fe1    	str	x1, [sp, #0x18]
100000c24: f85f03a8    	ldur	x8, [x29, #-0x10]
100000c28: d37be908    	lsl	x8, x8, #5
100000c2c: f9000be8    	str	x8, [sp, #0x10]
100000c30: f9400fe0    	ldr	x0, [sp, #0x18]
100000c34: 94000019    	bl	0x100000c98 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000c38: 36000120    	tbz	w0, #0x0, 0x100000c5c <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x4c>
100000c3c: 14000001    	b	0x100000c40 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x30>
100000c40: f9400fe8    	ldr	x8, [sp, #0x18]
100000c44: f90007e8    	str	x8, [sp, #0x8]
100000c48: f9400be0    	ldr	x0, [sp, #0x10]
100000c4c: f94007e1    	ldr	x1, [sp, #0x8]
100000c50: 94000019    	bl	0x100000cb4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>
100000c54: f81f83a0    	stur	x0, [x29, #-0x8]
100000c58: 14000005    	b	0x100000c6c <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000c5c: f9400be0    	ldr	x0, [sp, #0x10]
100000c60: 94000020    	bl	0x100000ce0 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>
100000c64: f81f83a0    	stur	x0, [x29, #-0x8]
100000c68: 14000001    	b	0x100000c6c <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000c6c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c70: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000c74: 910103ff    	add	sp, sp, #0x40
100000c78: d65f03c0    	ret

0000000100000c7c <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000c7c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000c80: 910003fd    	mov	x29, sp
100000c84: 94000003    	bl	0x100000c90 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000c88: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000c8c: d65f03c0    	ret

0000000100000c90 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000c90: 92800000    	mov	x0, #-0x1               ; =-1
100000c94: d65f03c0    	ret

0000000100000c98 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000c98: d10043ff    	sub	sp, sp, #0x10
100000c9c: f90007e0    	str	x0, [sp, #0x8]
100000ca0: f94007e8    	ldr	x8, [sp, #0x8]
100000ca4: f1004108    	subs	x8, x8, #0x10
100000ca8: 1a9f97e0    	cset	w0, hi
100000cac: 910043ff    	add	sp, sp, #0x10
100000cb0: d65f03c0    	ret

0000000100000cb4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>:
100000cb4: d10083ff    	sub	sp, sp, #0x20
100000cb8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000cbc: 910043fd    	add	x29, sp, #0x10
100000cc0: f90007e0    	str	x0, [sp, #0x8]
100000cc4: f90003e1    	str	x1, [sp]
100000cc8: f94007e0    	ldr	x0, [sp, #0x8]
100000ccc: f94003e1    	ldr	x1, [sp]
100000cd0: 940002ab    	bl	0x10000177c <___stack_chk_guard+0x10000177c>
100000cd4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cd8: 910083ff    	add	sp, sp, #0x20
100000cdc: d65f03c0    	ret

0000000100000ce0 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>:
100000ce0: d10083ff    	sub	sp, sp, #0x20
100000ce4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ce8: 910043fd    	add	x29, sp, #0x10
100000cec: f90007e0    	str	x0, [sp, #0x8]
100000cf0: f94007e0    	ldr	x0, [sp, #0x8]
100000cf4: 940002a5    	bl	0x100001788 <___stack_chk_guard+0x100001788>
100000cf8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cfc: 910083ff    	add	sp, sp, #0x20
100000d00: d65f03c0    	ret

0000000100000d04 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000d04: d10103ff    	sub	sp, sp, #0x40
100000d08: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000d0c: 9100c3fd    	add	x29, sp, #0x30
100000d10: f81f03a0    	stur	x0, [x29, #-0x10]
100000d14: f9000fe1    	str	x1, [sp, #0x18]
100000d18: f85f03a0    	ldur	x0, [x29, #-0x10]
100000d1c: f90003e0    	str	x0, [sp]
100000d20: d2800001    	mov	x1, #0x0                ; =0
100000d24: 94000027    	bl	0x100000dc0 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000d28: f94003e8    	ldr	x8, [sp]
100000d2c: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000d30: 91032129    	add	x9, x9, #0xc8
100000d34: 91004129    	add	x9, x9, #0x10
100000d38: f9000109    	str	x9, [x8]
100000d3c: 91006100    	add	x0, x8, #0x18
100000d40: d10007a1    	sub	x1, x29, #0x1
100000d44: 94000032    	bl	0x100000e0c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>
100000d48: 14000001    	b	0x100000d4c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x48>
100000d4c: f94003e0    	ldr	x0, [sp]
100000d50: 9400003c    	bl	0x100000e40 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000d54: f94003e0    	ldr	x0, [sp]
100000d58: 97ffff31    	bl	0x100000a1c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000d5c: aa0003e1    	mov	x1, x0
100000d60: f9400fe2    	ldr	x2, [sp, #0x18]
100000d64: 91002fe0    	add	x0, sp, #0xb
100000d68: 9400028b    	bl	0x100001794 <___stack_chk_guard+0x100001794>
100000d6c: 14000001    	b	0x100000d70 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x6c>
100000d70: f94003e0    	ldr	x0, [sp]
100000d74: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000d78: 910103ff    	add	sp, sp, #0x40
100000d7c: d65f03c0    	ret
100000d80: f9000be0    	str	x0, [sp, #0x10]
100000d84: aa0103e8    	mov	x8, x1
100000d88: b9000fe8    	str	w8, [sp, #0xc]
100000d8c: 14000008    	b	0x100000dac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000d90: f94003e8    	ldr	x8, [sp]
100000d94: f9000be0    	str	x0, [sp, #0x10]
100000d98: aa0103e9    	mov	x9, x1
100000d9c: b9000fe9    	str	w9, [sp, #0xc]
100000da0: 91006100    	add	x0, x8, #0x18
100000da4: 9400003d    	bl	0x100000e98 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000da8: 14000001    	b	0x100000dac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000dac: f94003e0    	ldr	x0, [sp]
100000db0: 9400027c    	bl	0x1000017a0 <___stack_chk_guard+0x1000017a0>
100000db4: 14000001    	b	0x100000db8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xb4>
100000db8: f9400be0    	ldr	x0, [sp, #0x10]
100000dbc: 94000258    	bl	0x10000171c <___stack_chk_guard+0x10000171c>

0000000100000dc0 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000dc0: d100c3ff    	sub	sp, sp, #0x30
100000dc4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000dc8: 910083fd    	add	x29, sp, #0x20
100000dcc: f81f83a0    	stur	x0, [x29, #-0x8]
100000dd0: f9000be1    	str	x1, [sp, #0x10]
100000dd4: f85f83a0    	ldur	x0, [x29, #-0x8]
100000dd8: f90007e0    	str	x0, [sp, #0x8]
100000ddc: f9400be1    	ldr	x1, [sp, #0x10]
100000de0: 94000070    	bl	0x100000fa0 <__ZNSt3__114__shared_countC2B8ne200100El>
100000de4: f94007e0    	ldr	x0, [sp, #0x8]
100000de8: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000dec: f9404508    	ldr	x8, [x8, #0x88]
100000df0: 91004108    	add	x8, x8, #0x10
100000df4: f9000008    	str	x8, [x0]
100000df8: f9400be8    	ldr	x8, [sp, #0x10]
100000dfc: f9000808    	str	x8, [x0, #0x10]
100000e00: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e04: 9100c3ff    	add	sp, sp, #0x30
100000e08: d65f03c0    	ret

0000000100000e0c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>:
100000e0c: d100c3ff    	sub	sp, sp, #0x30
100000e10: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e14: 910083fd    	add	x29, sp, #0x20
100000e18: f81f83a0    	stur	x0, [x29, #-0x8]
100000e1c: f9000be1    	str	x1, [sp, #0x10]
100000e20: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e24: f90007e0    	str	x0, [sp, #0x8]
100000e28: f9400be1    	ldr	x1, [sp, #0x10]
100000e2c: 94000069    	bl	0x100000fd0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>
100000e30: f94007e0    	ldr	x0, [sp, #0x8]
100000e34: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e38: 9100c3ff    	add	sp, sp, #0x30
100000e3c: d65f03c0    	ret

0000000100000e40 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>:
100000e40: d10083ff    	sub	sp, sp, #0x20
100000e44: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e48: 910043fd    	add	x29, sp, #0x10
100000e4c: f90007e0    	str	x0, [sp, #0x8]
100000e50: f94007e8    	ldr	x8, [sp, #0x8]
100000e54: 91006100    	add	x0, x8, #0x18
100000e58: 9400006a    	bl	0x100001000 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000e5c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e60: 910083ff    	add	sp, sp, #0x20
100000e64: d65f03c0    	ret

0000000100000e68 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE9constructB8ne200100IiJiEvLi0EEEvRS2_PT_DpOT0_>:
100000e68: d100c3ff    	sub	sp, sp, #0x30
100000e6c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e70: 910083fd    	add	x29, sp, #0x20
100000e74: f81f83a0    	stur	x0, [x29, #-0x8]
100000e78: f9000be1    	str	x1, [sp, #0x10]
100000e7c: f90007e2    	str	x2, [sp, #0x8]
100000e80: f9400be0    	ldr	x0, [sp, #0x10]
100000e84: f94007e1    	ldr	x1, [sp, #0x8]
100000e88: 94000063    	bl	0x100001014 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000e8c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e90: 9100c3ff    	add	sp, sp, #0x30
100000e94: d65f03c0    	ret

0000000100000e98 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>:
100000e98: d10083ff    	sub	sp, sp, #0x20
100000e9c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ea0: 910043fd    	add	x29, sp, #0x10
100000ea4: f90007e0    	str	x0, [sp, #0x8]
100000ea8: f94007e0    	ldr	x0, [sp, #0x8]
100000eac: f90003e0    	str	x0, [sp]
100000eb0: 9400006d    	bl	0x100001064 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>
100000eb4: f94003e0    	ldr	x0, [sp]
100000eb8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ebc: 910083ff    	add	sp, sp, #0x20
100000ec0: d65f03c0    	ret

0000000100000ec4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000ec4: d10083ff    	sub	sp, sp, #0x20
100000ec8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ecc: 910043fd    	add	x29, sp, #0x10
100000ed0: f90007e0    	str	x0, [sp, #0x8]
100000ed4: f94007e0    	ldr	x0, [sp, #0x8]
100000ed8: f90003e0    	str	x0, [sp]
100000edc: 9400006d    	bl	0x100001090 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>
100000ee0: f94003e0    	ldr	x0, [sp]
100000ee4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ee8: 910083ff    	add	sp, sp, #0x20
100000eec: d65f03c0    	ret

0000000100000ef0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000ef0: d10083ff    	sub	sp, sp, #0x20
100000ef4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ef8: 910043fd    	add	x29, sp, #0x10
100000efc: f90007e0    	str	x0, [sp, #0x8]
100000f00: f94007e0    	ldr	x0, [sp, #0x8]
100000f04: f90003e0    	str	x0, [sp]
100000f08: 97ffffef    	bl	0x100000ec4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>
100000f0c: f94003e0    	ldr	x0, [sp]
100000f10: 94000227    	bl	0x1000017ac <___stack_chk_guard+0x1000017ac>
100000f14: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f18: 910083ff    	add	sp, sp, #0x20
100000f1c: d65f03c0    	ret

0000000100000f20 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000f20: d10083ff    	sub	sp, sp, #0x20
100000f24: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f28: 910043fd    	add	x29, sp, #0x10
100000f2c: f90007e0    	str	x0, [sp, #0x8]
100000f30: f94007e0    	ldr	x0, [sp, #0x8]
100000f34: 94000221    	bl	0x1000017b8 <___stack_chk_guard+0x1000017b8>
100000f38: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f3c: 910083ff    	add	sp, sp, #0x20
100000f40: d65f03c0    	ret

0000000100000f44 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000f44: d100c3ff    	sub	sp, sp, #0x30
100000f48: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000f4c: 910083fd    	add	x29, sp, #0x20
100000f50: f81f83a0    	stur	x0, [x29, #-0x8]
100000f54: f85f83a0    	ldur	x0, [x29, #-0x8]
100000f58: f90003e0    	str	x0, [sp]
100000f5c: 97ffffb9    	bl	0x100000e40 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000f60: aa0003e1    	mov	x1, x0
100000f64: d10027a0    	sub	x0, x29, #0x9
100000f68: f90007e0    	str	x0, [sp, #0x8]
100000f6c: 97fffed5    	bl	0x100000ac0 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000f70: f94003e8    	ldr	x8, [sp]
100000f74: 91006100    	add	x0, x8, #0x18
100000f78: 97ffffc8    	bl	0x100000e98 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000f7c: f94003e0    	ldr	x0, [sp]
100000f80: 94000086    	bl	0x100001198 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>
100000f84: aa0003e1    	mov	x1, x0
100000f88: f94007e0    	ldr	x0, [sp, #0x8]
100000f8c: d2800022    	mov	x2, #0x1                ; =1
100000f90: 94000075    	bl	0x100001164 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100000f94: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000f98: 9100c3ff    	add	sp, sp, #0x30
100000f9c: d65f03c0    	ret

0000000100000fa0 <__ZNSt3__114__shared_countC2B8ne200100El>:
100000fa0: d10043ff    	sub	sp, sp, #0x10
100000fa4: f90007e0    	str	x0, [sp, #0x8]
100000fa8: f90003e1    	str	x1, [sp]
100000fac: f94007e0    	ldr	x0, [sp, #0x8]
100000fb0: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000fb4: f9405108    	ldr	x8, [x8, #0xa0]
100000fb8: 91004108    	add	x8, x8, #0x10
100000fbc: f9000008    	str	x8, [x0]
100000fc0: f94003e8    	ldr	x8, [sp]
100000fc4: f9000408    	str	x8, [x0, #0x8]
100000fc8: 910043ff    	add	sp, sp, #0x10
100000fcc: d65f03c0    	ret

0000000100000fd0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>:
100000fd0: d100c3ff    	sub	sp, sp, #0x30
100000fd4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000fd8: 910083fd    	add	x29, sp, #0x20
100000fdc: f81f83a0    	stur	x0, [x29, #-0x8]
100000fe0: f9000be1    	str	x1, [sp, #0x10]
100000fe4: f85f83a0    	ldur	x0, [x29, #-0x8]
100000fe8: f90007e0    	str	x0, [sp, #0x8]
100000fec: 94000005    	bl	0x100001000 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000ff0: f94007e0    	ldr	x0, [sp, #0x8]
100000ff4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ff8: 9100c3ff    	add	sp, sp, #0x30
100000ffc: d65f03c0    	ret

0000000100001000 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>:
100001000: d10043ff    	sub	sp, sp, #0x10
100001004: f90007e0    	str	x0, [sp, #0x8]
100001008: f94007e0    	ldr	x0, [sp, #0x8]
10000100c: 910043ff    	add	sp, sp, #0x10
100001010: d65f03c0    	ret

0000000100001014 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100001014: d10083ff    	sub	sp, sp, #0x20
100001018: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000101c: 910043fd    	add	x29, sp, #0x10
100001020: f90007e0    	str	x0, [sp, #0x8]
100001024: f90003e1    	str	x1, [sp]
100001028: f94007e0    	ldr	x0, [sp, #0x8]
10000102c: f94003e1    	ldr	x1, [sp]
100001030: 94000004    	bl	0x100001040 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100001034: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001038: 910083ff    	add	sp, sp, #0x20
10000103c: d65f03c0    	ret

0000000100001040 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100001040: d10043ff    	sub	sp, sp, #0x10
100001044: f90007e0    	str	x0, [sp, #0x8]
100001048: f90003e1    	str	x1, [sp]
10000104c: f94007e0    	ldr	x0, [sp, #0x8]
100001050: f94003e8    	ldr	x8, [sp]
100001054: b9400108    	ldr	w8, [x8]
100001058: b9000008    	str	w8, [x0]
10000105c: 910043ff    	add	sp, sp, #0x10
100001060: d65f03c0    	ret

0000000100001064 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>:
100001064: d10083ff    	sub	sp, sp, #0x20
100001068: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000106c: 910043fd    	add	x29, sp, #0x10
100001070: f90007e0    	str	x0, [sp, #0x8]
100001074: f94007e0    	ldr	x0, [sp, #0x8]
100001078: f90003e0    	str	x0, [sp]
10000107c: 97ffffe1    	bl	0x100001000 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100001080: f94003e0    	ldr	x0, [sp]
100001084: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001088: 910083ff    	add	sp, sp, #0x20
10000108c: d65f03c0    	ret

0000000100001090 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>:
100001090: d10083ff    	sub	sp, sp, #0x20
100001094: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001098: 910043fd    	add	x29, sp, #0x10
10000109c: f90007e0    	str	x0, [sp, #0x8]
1000010a0: f94007e8    	ldr	x8, [sp, #0x8]
1000010a4: f90003e8    	str	x8, [sp]
1000010a8: f0000009    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000010ac: 91032129    	add	x9, x9, #0xc8
1000010b0: 91004129    	add	x9, x9, #0x10
1000010b4: f9000109    	str	x9, [x8]
1000010b8: 91006100    	add	x0, x8, #0x18
1000010bc: 97ffff77    	bl	0x100000e98 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
1000010c0: f94003e0    	ldr	x0, [sp]
1000010c4: 940001b7    	bl	0x1000017a0 <___stack_chk_guard+0x1000017a0>
1000010c8: f94003e0    	ldr	x0, [sp]
1000010cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000010d0: 910083ff    	add	sp, sp, #0x20
1000010d4: d65f03c0    	ret

00000001000010d8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv>:
1000010d8: d100c3ff    	sub	sp, sp, #0x30
1000010dc: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000010e0: 910083fd    	add	x29, sp, #0x20
1000010e4: f81f83a0    	stur	x0, [x29, #-0x8]
1000010e8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000010ec: f90007e0    	str	x0, [sp, #0x8]
1000010f0: 97ffff54    	bl	0x100000e40 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
1000010f4: f94007e0    	ldr	x0, [sp, #0x8]
1000010f8: 97fffe49    	bl	0x100000a1c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
1000010fc: aa0003e1    	mov	x1, x0
100001100: d10027a0    	sub	x0, x29, #0x9
100001104: 940001b0    	bl	0x1000017c4 <___stack_chk_guard+0x1000017c4>
100001108: 14000001    	b	0x10000110c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv+0x34>
10000110c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001110: 9100c3ff    	add	sp, sp, #0x30
100001114: d65f03c0    	ret
100001118: 9400000b    	bl	0x100001144 <___clang_call_terminate>

000000010000111c <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE7destroyB8ne200100IivLi0EEEvRS2_PT_>:
10000111c: d10083ff    	sub	sp, sp, #0x20
100001120: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001124: 910043fd    	add	x29, sp, #0x10
100001128: f90007e0    	str	x0, [sp, #0x8]
10000112c: f90003e1    	str	x1, [sp]
100001130: f94003e0    	ldr	x0, [sp]
100001134: 94000008    	bl	0x100001154 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>
100001138: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000113c: 910083ff    	add	sp, sp, #0x20
100001140: d65f03c0    	ret

0000000100001144 <___clang_call_terminate>:
100001144: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100001148: 910003fd    	mov	x29, sp
10000114c: 940001a1    	bl	0x1000017d0 <___stack_chk_guard+0x1000017d0>
100001150: 940001a3    	bl	0x1000017dc <___stack_chk_guard+0x1000017dc>

0000000100001154 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>:
100001154: d10043ff    	sub	sp, sp, #0x10
100001158: f90007e0    	str	x0, [sp, #0x8]
10000115c: 910043ff    	add	sp, sp, #0x10
100001160: d65f03c0    	ret

0000000100001164 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>:
100001164: d100c3ff    	sub	sp, sp, #0x30
100001168: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000116c: 910083fd    	add	x29, sp, #0x20
100001170: f81f83a0    	stur	x0, [x29, #-0x8]
100001174: f9000be1    	str	x1, [sp, #0x10]
100001178: f90007e2    	str	x2, [sp, #0x8]
10000117c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001180: f9400be1    	ldr	x1, [sp, #0x10]
100001184: f94007e2    	ldr	x2, [sp, #0x8]
100001188: 94000009    	bl	0x1000011ac <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>
10000118c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001190: 9100c3ff    	add	sp, sp, #0x30
100001194: d65f03c0    	ret

0000000100001198 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>:
100001198: d10043ff    	sub	sp, sp, #0x10
10000119c: f90007e0    	str	x0, [sp, #0x8]
1000011a0: f94007e0    	ldr	x0, [sp, #0x8]
1000011a4: 910043ff    	add	sp, sp, #0x10
1000011a8: d65f03c0    	ret

00000001000011ac <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>:
1000011ac: d100c3ff    	sub	sp, sp, #0x30
1000011b0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000011b4: 910083fd    	add	x29, sp, #0x20
1000011b8: f81f83a0    	stur	x0, [x29, #-0x8]
1000011bc: f9000be1    	str	x1, [sp, #0x10]
1000011c0: f90007e2    	str	x2, [sp, #0x8]
1000011c4: f9400be0    	ldr	x0, [sp, #0x10]
1000011c8: f94007e1    	ldr	x1, [sp, #0x8]
1000011cc: d2800102    	mov	x2, #0x8                ; =8
1000011d0: 94000004    	bl	0x1000011e0 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
1000011d4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000011d8: 9100c3ff    	add	sp, sp, #0x30
1000011dc: d65f03c0    	ret

00000001000011e0 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
1000011e0: d10103ff    	sub	sp, sp, #0x40
1000011e4: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000011e8: 9100c3fd    	add	x29, sp, #0x30
1000011ec: f81f83a0    	stur	x0, [x29, #-0x8]
1000011f0: f81f03a1    	stur	x1, [x29, #-0x10]
1000011f4: f9000fe2    	str	x2, [sp, #0x18]
1000011f8: f85f03a8    	ldur	x8, [x29, #-0x10]
1000011fc: d37be908    	lsl	x8, x8, #5
100001200: f9000be8    	str	x8, [sp, #0x10]
100001204: f9400fe0    	ldr	x0, [sp, #0x18]
100001208: 97fffea4    	bl	0x100000c98 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
10000120c: 36000100    	tbz	w0, #0x0, 0x10000122c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x4c>
100001210: 14000001    	b	0x100001214 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x34>
100001214: f9400fe8    	ldr	x8, [sp, #0x18]
100001218: f90007e8    	str	x8, [sp, #0x8]
10000121c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001220: f94007e1    	ldr	x1, [sp, #0x8]
100001224: 94000008    	bl	0x100001244 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>
100001228: 14000004    	b	0x100001238 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
10000122c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001230: 94000010    	bl	0x100001270 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>
100001234: 14000001    	b	0x100001238 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
100001238: a9437bfd    	ldp	x29, x30, [sp, #0x30]
10000123c: 910103ff    	add	sp, sp, #0x40
100001240: d65f03c0    	ret

0000000100001244 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>:
100001244: d10083ff    	sub	sp, sp, #0x20
100001248: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000124c: 910043fd    	add	x29, sp, #0x10
100001250: f90007e0    	str	x0, [sp, #0x8]
100001254: f90003e1    	str	x1, [sp]
100001258: f94007e0    	ldr	x0, [sp, #0x8]
10000125c: f94003e1    	ldr	x1, [sp]
100001260: 94000162    	bl	0x1000017e8 <___stack_chk_guard+0x1000017e8>
100001264: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001268: 910083ff    	add	sp, sp, #0x20
10000126c: d65f03c0    	ret

0000000100001270 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>:
100001270: d10083ff    	sub	sp, sp, #0x20
100001274: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001278: 910043fd    	add	x29, sp, #0x10
10000127c: f90007e0    	str	x0, [sp, #0x8]
100001280: f94007e0    	ldr	x0, [sp, #0x8]
100001284: 9400014a    	bl	0x1000017ac <___stack_chk_guard+0x1000017ac>
100001288: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000128c: 910083ff    	add	sp, sp, #0x20
100001290: d65f03c0    	ret

0000000100001294 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>:
100001294: d10083ff    	sub	sp, sp, #0x20
100001298: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000129c: 910043fd    	add	x29, sp, #0x10
1000012a0: f90007e0    	str	x0, [sp, #0x8]
1000012a4: f94007e0    	ldr	x0, [sp, #0x8]
1000012a8: f90003e0    	str	x0, [sp]
1000012ac: 94000009    	bl	0x1000012d0 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>
1000012b0: f94003e0    	ldr	x0, [sp]
1000012b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012b8: 910083ff    	add	sp, sp, #0x20
1000012bc: d65f03c0    	ret

00000001000012c0 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>:
1000012c0: d10043ff    	sub	sp, sp, #0x10
1000012c4: f90007e0    	str	x0, [sp, #0x8]
1000012c8: 910043ff    	add	sp, sp, #0x10
1000012cc: d65f03c0    	ret

00000001000012d0 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>:
1000012d0: d10043ff    	sub	sp, sp, #0x10
1000012d4: f90007e0    	str	x0, [sp, #0x8]
1000012d8: f94007e0    	ldr	x0, [sp, #0x8]
1000012dc: f900001f    	str	xzr, [x0]
1000012e0: f900041f    	str	xzr, [x0, #0x8]
1000012e4: 910043ff    	add	sp, sp, #0x10
1000012e8: d65f03c0    	ret

00000001000012ec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>:
1000012ec: d10043ff    	sub	sp, sp, #0x10
1000012f0: f90007e0    	str	x0, [sp, #0x8]
1000012f4: f94007e0    	ldr	x0, [sp, #0x8]
1000012f8: 910043ff    	add	sp, sp, #0x10
1000012fc: d65f03c0    	ret

0000000100001300 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>:
100001300: d10083ff    	sub	sp, sp, #0x20
100001304: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001308: 910043fd    	add	x29, sp, #0x10
10000130c: f90007e0    	str	x0, [sp, #0x8]
100001310: f94007e0    	ldr	x0, [sp, #0x8]
100001314: f90003e0    	str	x0, [sp]
100001318: 94000005    	bl	0x10000132c <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>
10000131c: f94003e0    	ldr	x0, [sp]
100001320: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001324: 910083ff    	add	sp, sp, #0x20
100001328: d65f03c0    	ret

000000010000132c <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>:
10000132c: d10083ff    	sub	sp, sp, #0x20
100001330: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001334: 910043fd    	add	x29, sp, #0x10
100001338: f90007e0    	str	x0, [sp, #0x8]
10000133c: f94007e8    	ldr	x8, [sp, #0x8]
100001340: f90003e8    	str	x8, [sp]
100001344: f9400908    	ldr	x8, [x8, #0x10]
100001348: b40000e8    	cbz	x8, 0x100001364 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
10000134c: 14000001    	b	0x100001350 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x24>
100001350: f94003e0    	ldr	x0, [sp]
100001354: f9400801    	ldr	x1, [x0, #0x10]
100001358: f9400402    	ldr	x2, [x0, #0x8]
10000135c: 97ffff82    	bl	0x100001164 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100001360: 14000001    	b	0x100001364 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
100001364: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001368: 910083ff    	add	sp, sp, #0x20
10000136c: d65f03c0    	ret

0000000100001370 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>:
100001370: d10083ff    	sub	sp, sp, #0x20
100001374: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001378: 910043fd    	add	x29, sp, #0x10
10000137c: f90007e0    	str	x0, [sp, #0x8]
100001380: f94007e0    	ldr	x0, [sp, #0x8]
100001384: f90003e0    	str	x0, [sp]
100001388: 94000005    	bl	0x10000139c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
10000138c: f94003e0    	ldr	x0, [sp]
100001390: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001394: 910083ff    	add	sp, sp, #0x20
100001398: d65f03c0    	ret

000000010000139c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
10000139c: d10043ff    	sub	sp, sp, #0x10
1000013a0: f90007e0    	str	x0, [sp, #0x8]
1000013a4: f94007e0    	ldr	x0, [sp, #0x8]
1000013a8: 910043ff    	add	sp, sp, #0x10
1000013ac: d65f03c0    	ret

00000001000013b0 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>:
1000013b0: d100c3ff    	sub	sp, sp, #0x30
1000013b4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000013b8: 910083fd    	add	x29, sp, #0x20
1000013bc: f9000be0    	str	x0, [sp, #0x10]
1000013c0: f9400be8    	ldr	x8, [sp, #0x10]
1000013c4: f90007e8    	str	x8, [sp, #0x8]
1000013c8: aa0803e9    	mov	x9, x8
1000013cc: f81f83a9    	stur	x9, [x29, #-0x8]
1000013d0: f9400508    	ldr	x8, [x8, #0x8]
1000013d4: b40000c8    	cbz	x8, 0x1000013ec <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
1000013d8: 14000001    	b	0x1000013dc <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x2c>
1000013dc: f94007e8    	ldr	x8, [sp, #0x8]
1000013e0: f9400500    	ldr	x0, [x8, #0x8]
1000013e4: 94000006    	bl	0x1000013fc <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
1000013e8: 14000001    	b	0x1000013ec <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
1000013ec: f85f83a0    	ldur	x0, [x29, #-0x8]
1000013f0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000013f4: 9100c3ff    	add	sp, sp, #0x30
1000013f8: d65f03c0    	ret

00000001000013fc <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
1000013fc: d10083ff    	sub	sp, sp, #0x20
100001400: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001404: 910043fd    	add	x29, sp, #0x10
100001408: f90007e0    	str	x0, [sp, #0x8]
10000140c: f94007e0    	ldr	x0, [sp, #0x8]
100001410: f90003e0    	str	x0, [sp]
100001414: 94000009    	bl	0x100001438 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
100001418: 360000a0    	tbz	w0, #0x0, 0x10000142c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
10000141c: 14000001    	b	0x100001420 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
100001420: f94003e0    	ldr	x0, [sp]
100001424: 940000c1    	bl	0x100001728 <___stack_chk_guard+0x100001728>
100001428: 14000001    	b	0x10000142c <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
10000142c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001430: 910083ff    	add	sp, sp, #0x20
100001434: d65f03c0    	ret

0000000100001438 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
100001438: d100c3ff    	sub	sp, sp, #0x30
10000143c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001440: 910083fd    	add	x29, sp, #0x20
100001444: f9000be0    	str	x0, [sp, #0x10]
100001448: f9400be8    	ldr	x8, [sp, #0x10]
10000144c: f90007e8    	str	x8, [sp, #0x8]
100001450: 91002100    	add	x0, x8, #0x8
100001454: 94000017    	bl	0x1000014b0 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
100001458: b1000408    	adds	x8, x0, #0x1
10000145c: 54000161    	b.ne	0x100001488 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
100001460: 14000001    	b	0x100001464 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
100001464: f94007e0    	ldr	x0, [sp, #0x8]
100001468: f9400008    	ldr	x8, [x0]
10000146c: f9400908    	ldr	x8, [x8, #0x10]
100001470: d63f0100    	blr	x8
100001474: 52800028    	mov	w8, #0x1                ; =1
100001478: 12000108    	and	w8, w8, #0x1
10000147c: 12000108    	and	w8, w8, #0x1
100001480: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001484: 14000006    	b	0x10000149c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100001488: 52800008    	mov	w8, #0x0                ; =0
10000148c: 12000108    	and	w8, w8, #0x1
100001490: 12000108    	and	w8, w8, #0x1
100001494: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001498: 14000001    	b	0x10000149c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
10000149c: 385ff3a8    	ldurb	w8, [x29, #-0x1]
1000014a0: 12000100    	and	w0, w8, #0x1
1000014a4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000014a8: 9100c3ff    	add	sp, sp, #0x30
1000014ac: d65f03c0    	ret

00000001000014b0 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
1000014b0: d10083ff    	sub	sp, sp, #0x20
1000014b4: f9000fe0    	str	x0, [sp, #0x18]
1000014b8: f9400fe8    	ldr	x8, [sp, #0x18]
1000014bc: 92800009    	mov	x9, #-0x1               ; =-1
1000014c0: f9000be9    	str	x9, [sp, #0x10]
1000014c4: f9400be9    	ldr	x9, [sp, #0x10]
1000014c8: f8e90108    	ldaddal	x9, x8, [x8]
1000014cc: 8b090108    	add	x8, x8, x9
1000014d0: f90007e8    	str	x8, [sp, #0x8]
1000014d4: f94007e0    	ldr	x0, [sp, #0x8]
1000014d8: 910083ff    	add	sp, sp, #0x20
1000014dc: d65f03c0    	ret

00000001000014e0 <__ZNSt3__18weak_ptrIiEC1IiLi0EEERKNS_10shared_ptrIT_EE>:
1000014e0: d100c3ff    	sub	sp, sp, #0x30
1000014e4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000014e8: 910083fd    	add	x29, sp, #0x20
1000014ec: f81f83a0    	stur	x0, [x29, #-0x8]
1000014f0: f9000be1    	str	x1, [sp, #0x10]
1000014f4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000014f8: f90007e0    	str	x0, [sp, #0x8]
1000014fc: f9400be1    	ldr	x1, [sp, #0x10]
100001500: 94000016    	bl	0x100001558 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE>
100001504: f94007e0    	ldr	x0, [sp, #0x8]
100001508: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000150c: 9100c3ff    	add	sp, sp, #0x30
100001510: d65f03c0    	ret

0000000100001514 <__ZNSt3__18weak_ptrIiE4swapERS1_>:
100001514: d100c3ff    	sub	sp, sp, #0x30
100001518: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000151c: 910083fd    	add	x29, sp, #0x20
100001520: f81f83a0    	stur	x0, [x29, #-0x8]
100001524: f9000be1    	str	x1, [sp, #0x10]
100001528: f85f83a0    	ldur	x0, [x29, #-0x8]
10000152c: f90007e0    	str	x0, [sp, #0x8]
100001530: f9400be1    	ldr	x1, [sp, #0x10]
100001534: 94000039    	bl	0x100001618 <__ZNSt3__14swapB8ne200100IPiEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS3_EE5valueEvE4typeERS3_S6_>
100001538: f94007e9    	ldr	x9, [sp, #0x8]
10000153c: f9400be8    	ldr	x8, [sp, #0x10]
100001540: 91002120    	add	x0, x9, #0x8
100001544: 91002101    	add	x1, x8, #0x8
100001548: 94000043    	bl	0x100001654 <__ZNSt3__14swapB8ne200100IPNS_19__shared_weak_countEEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS4_EE5valueEvE4typeERS4_S7_>
10000154c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001550: 9100c3ff    	add	sp, sp, #0x30
100001554: d65f03c0    	ret

0000000100001558 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE>:
100001558: d100c3ff    	sub	sp, sp, #0x30
10000155c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001560: 910083fd    	add	x29, sp, #0x20
100001564: f9000be0    	str	x0, [sp, #0x10]
100001568: f90007e1    	str	x1, [sp, #0x8]
10000156c: f9400be8    	ldr	x8, [sp, #0x10]
100001570: f90003e8    	str	x8, [sp]
100001574: aa0803e9    	mov	x9, x8
100001578: f81f83a9    	stur	x9, [x29, #-0x8]
10000157c: f94007e9    	ldr	x9, [sp, #0x8]
100001580: f9400129    	ldr	x9, [x9]
100001584: f9000109    	str	x9, [x8]
100001588: f94007e9    	ldr	x9, [sp, #0x8]
10000158c: f9400529    	ldr	x9, [x9, #0x8]
100001590: f9000509    	str	x9, [x8, #0x8]
100001594: f9400508    	ldr	x8, [x8, #0x8]
100001598: b40000c8    	cbz	x8, 0x1000015b0 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE+0x58>
10000159c: 14000001    	b	0x1000015a0 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE+0x48>
1000015a0: f94003e8    	ldr	x8, [sp]
1000015a4: f9400500    	ldr	x0, [x8, #0x8]
1000015a8: 94000006    	bl	0x1000015c0 <__ZNSt3__119__shared_weak_count10__add_weakB8ne200100Ev>
1000015ac: 14000001    	b	0x1000015b0 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE+0x58>
1000015b0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000015b4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000015b8: 9100c3ff    	add	sp, sp, #0x30
1000015bc: d65f03c0    	ret

00000001000015c0 <__ZNSt3__119__shared_weak_count10__add_weakB8ne200100Ev>:
1000015c0: d10083ff    	sub	sp, sp, #0x20
1000015c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000015c8: 910043fd    	add	x29, sp, #0x10
1000015cc: f90007e0    	str	x0, [sp, #0x8]
1000015d0: f94007e8    	ldr	x8, [sp, #0x8]
1000015d4: 91004100    	add	x0, x8, #0x10
1000015d8: 94000004    	bl	0x1000015e8 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>
1000015dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000015e0: 910083ff    	add	sp, sp, #0x20
1000015e4: d65f03c0    	ret

00000001000015e8 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>:
1000015e8: d10083ff    	sub	sp, sp, #0x20
1000015ec: f9000fe0    	str	x0, [sp, #0x18]
1000015f0: f9400fe8    	ldr	x8, [sp, #0x18]
1000015f4: d2800029    	mov	x9, #0x1                ; =1
1000015f8: f9000be9    	str	x9, [sp, #0x10]
1000015fc: f9400be9    	ldr	x9, [sp, #0x10]
100001600: f8290108    	ldadd	x9, x8, [x8]
100001604: 8b090108    	add	x8, x8, x9
100001608: f90007e8    	str	x8, [sp, #0x8]
10000160c: f94007e0    	ldr	x0, [sp, #0x8]
100001610: 910083ff    	add	sp, sp, #0x20
100001614: d65f03c0    	ret

0000000100001618 <__ZNSt3__14swapB8ne200100IPiEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS3_EE5valueEvE4typeERS3_S6_>:
100001618: d10083ff    	sub	sp, sp, #0x20
10000161c: f9000fe0    	str	x0, [sp, #0x18]
100001620: f9000be1    	str	x1, [sp, #0x10]
100001624: f9400fe8    	ldr	x8, [sp, #0x18]
100001628: f9400108    	ldr	x8, [x8]
10000162c: f90007e8    	str	x8, [sp, #0x8]
100001630: f9400be8    	ldr	x8, [sp, #0x10]
100001634: f9400108    	ldr	x8, [x8]
100001638: f9400fe9    	ldr	x9, [sp, #0x18]
10000163c: f9000128    	str	x8, [x9]
100001640: f94007e8    	ldr	x8, [sp, #0x8]
100001644: f9400be9    	ldr	x9, [sp, #0x10]
100001648: f9000128    	str	x8, [x9]
10000164c: 910083ff    	add	sp, sp, #0x20
100001650: d65f03c0    	ret

0000000100001654 <__ZNSt3__14swapB8ne200100IPNS_19__shared_weak_countEEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS4_EE5valueEvE4typeERS4_S7_>:
100001654: d10083ff    	sub	sp, sp, #0x20
100001658: f9000fe0    	str	x0, [sp, #0x18]
10000165c: f9000be1    	str	x1, [sp, #0x10]
100001660: f9400fe8    	ldr	x8, [sp, #0x18]
100001664: f9400108    	ldr	x8, [x8]
100001668: f90007e8    	str	x8, [sp, #0x8]
10000166c: f9400be8    	ldr	x8, [sp, #0x10]
100001670: f9400108    	ldr	x8, [x8]
100001674: f9400fe9    	ldr	x9, [sp, #0x18]
100001678: f9000128    	str	x8, [x9]
10000167c: f94007e8    	ldr	x8, [sp, #0x8]
100001680: f9400be9    	ldr	x9, [sp, #0x10]
100001684: f9000128    	str	x8, [x9]
100001688: 910083ff    	add	sp, sp, #0x20
10000168c: d65f03c0    	ret

0000000100001690 <__ZNKSt3__119__shared_weak_count9use_countB8ne200100Ev>:
100001690: d10083ff    	sub	sp, sp, #0x20
100001694: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001698: 910043fd    	add	x29, sp, #0x10
10000169c: f90007e0    	str	x0, [sp, #0x8]
1000016a0: f94007e0    	ldr	x0, [sp, #0x8]
1000016a4: 94000004    	bl	0x1000016b4 <__ZNKSt3__114__shared_count9use_countB8ne200100Ev>
1000016a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000016ac: 910083ff    	add	sp, sp, #0x20
1000016b0: d65f03c0    	ret

00000001000016b4 <__ZNKSt3__114__shared_count9use_countB8ne200100Ev>:
1000016b4: d10083ff    	sub	sp, sp, #0x20
1000016b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000016bc: 910043fd    	add	x29, sp, #0x10
1000016c0: f90007e0    	str	x0, [sp, #0x8]
1000016c4: f94007e8    	ldr	x8, [sp, #0x8]
1000016c8: 91002100    	add	x0, x8, #0x8
1000016cc: 94000009    	bl	0x1000016f0 <__ZNSt3__121__libcpp_relaxed_loadB8ne200100IlEET_PKS1_>
1000016d0: f90003e0    	str	x0, [sp]
1000016d4: 14000001    	b	0x1000016d8 <__ZNKSt3__114__shared_count9use_countB8ne200100Ev+0x24>
1000016d8: f94003e8    	ldr	x8, [sp]
1000016dc: 91000500    	add	x0, x8, #0x1
1000016e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000016e4: 910083ff    	add	sp, sp, #0x20
1000016e8: d65f03c0    	ret
1000016ec: 97fffe96    	bl	0x100001144 <___clang_call_terminate>

00000001000016f0 <__ZNSt3__121__libcpp_relaxed_loadB8ne200100IlEET_PKS1_>:
1000016f0: d10043ff    	sub	sp, sp, #0x10
1000016f4: f90007e0    	str	x0, [sp, #0x8]
1000016f8: f94007e8    	ldr	x8, [sp, #0x8]
1000016fc: f9400108    	ldr	x8, [x8]
100001700: f90003e8    	str	x8, [sp]
100001704: f94003e0    	ldr	x0, [sp]
100001708: 910043ff    	add	sp, sp, #0x10
10000170c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001710 <__stubs>:
100001710: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001714: f9400610    	ldr	x16, [x16, #0x8]
100001718: d61f0200    	br	x16
10000171c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001720: f9400a10    	ldr	x16, [x16, #0x10]
100001724: d61f0200    	br	x16
100001728: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000172c: f9400e10    	ldr	x16, [x16, #0x18]
100001730: d61f0200    	br	x16
100001734: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001738: f9401610    	ldr	x16, [x16, #0x28]
10000173c: d61f0200    	br	x16
100001740: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001744: f9401a10    	ldr	x16, [x16, #0x30]
100001748: d61f0200    	br	x16
10000174c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001750: f9401e10    	ldr	x16, [x16, #0x38]
100001754: d61f0200    	br	x16
100001758: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000175c: f9402210    	ldr	x16, [x16, #0x40]
100001760: d61f0200    	br	x16
100001764: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001768: f9402610    	ldr	x16, [x16, #0x48]
10000176c: d61f0200    	br	x16
100001770: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001774: f9403210    	ldr	x16, [x16, #0x60]
100001778: d61f0200    	br	x16
10000177c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001780: f9403610    	ldr	x16, [x16, #0x68]
100001784: d61f0200    	br	x16
100001788: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000178c: f9403a10    	ldr	x16, [x16, #0x70]
100001790: d61f0200    	br	x16
100001794: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001798: f9403e10    	ldr	x16, [x16, #0x78]
10000179c: d61f0200    	br	x16
1000017a0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017a4: f9404210    	ldr	x16, [x16, #0x80]
1000017a8: d61f0200    	br	x16
1000017ac: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017b0: f9404a10    	ldr	x16, [x16, #0x90]
1000017b4: d61f0200    	br	x16
1000017b8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017bc: f9404e10    	ldr	x16, [x16, #0x98]
1000017c0: d61f0200    	br	x16
1000017c4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017c8: f9405610    	ldr	x16, [x16, #0xa8]
1000017cc: d61f0200    	br	x16
1000017d0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017d4: f9405a10    	ldr	x16, [x16, #0xb0]
1000017d8: d61f0200    	br	x16
1000017dc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017e0: f9405e10    	ldr	x16, [x16, #0xb8]
1000017e4: d61f0200    	br	x16
1000017e8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000017ec: f9406210    	ldr	x16, [x16, #0xc0]
1000017f0: d61f0200    	br	x16
