
/Users/jim/work/cppfort/micro-tests/results/memory/mem065-weak-ptr/mem065-weak-ptr_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z13test_weak_ptrv>:
100000538: d101c3ff    	sub	sp, sp, #0x70
10000053c: a9067bfd    	stp	x29, x30, [sp, #0x60]
100000540: 910183fd    	add	x29, sp, #0x60
100000544: d10053a0    	sub	x0, x29, #0x14
100000548: 52800548    	mov	w8, #0x2a               ; =42
10000054c: b81ec3a8    	stur	w8, [x29, #-0x14]
100000550: d10043a8    	sub	x8, x29, #0x10
100000554: f9000be8    	str	x8, [sp, #0x10]
100000558: 94000021    	bl	0x1000005dc <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>
10000055c: f9400be1    	ldr	x1, [sp, #0x10]
100000560: d100a3a0    	sub	x0, x29, #0x28
100000564: f9000fe0    	str	x0, [sp, #0x18]
100000568: 9400002d    	bl	0x10000061c <__ZNSt3__18weak_ptrIiEC1IiLi0EEERKNS_10shared_ptrIT_EE>
10000056c: f9400fe0    	ldr	x0, [sp, #0x18]
100000570: 9100a3e8    	add	x8, sp, #0x28
100000574: f90013e8    	str	x8, [sp, #0x20]
100000578: 94000036    	bl	0x100000650 <__ZNKSt3__18weak_ptrIiE4lockEv>
10000057c: f94013e0    	ldr	x0, [sp, #0x20]
100000580: 94000069    	bl	0x100000724 <__ZNKSt3__110shared_ptrIiEcvbB8ne200100Ev>
100000584: 360000e0    	tbz	w0, #0x0, 0x1000005a0 <__Z13test_weak_ptrv+0x68>
100000588: 14000001    	b	0x10000058c <__Z13test_weak_ptrv+0x54>
10000058c: 9100a3e0    	add	x0, sp, #0x28
100000590: 94000070    	bl	0x100000750 <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>
100000594: b9400008    	ldr	w8, [x0]
100000598: b9000fe8    	str	w8, [sp, #0xc]
10000059c: 14000004    	b	0x1000005ac <__Z13test_weak_ptrv+0x74>
1000005a0: 12800008    	mov	w8, #-0x1               ; =-1
1000005a4: b9000fe8    	str	w8, [sp, #0xc]
1000005a8: 14000001    	b	0x1000005ac <__Z13test_weak_ptrv+0x74>
1000005ac: b9400fe8    	ldr	w8, [sp, #0xc]
1000005b0: b9000be8    	str	w8, [sp, #0x8]
1000005b4: 9100a3e0    	add	x0, sp, #0x28
1000005b8: 9400006c    	bl	0x100000768 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
1000005bc: d100a3a0    	sub	x0, x29, #0x28
1000005c0: 94000075    	bl	0x100000794 <__ZNSt3__18weak_ptrIiED1Ev>
1000005c4: d10043a0    	sub	x0, x29, #0x10
1000005c8: 94000068    	bl	0x100000768 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
1000005cc: b9400be0    	ldr	w0, [sp, #0x8]
1000005d0: a9467bfd    	ldp	x29, x30, [sp, #0x60]
1000005d4: 9101c3ff    	add	sp, sp, #0x70
1000005d8: d65f03c0    	ret

00000001000005dc <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>:
1000005dc: d10103ff    	sub	sp, sp, #0x40
1000005e0: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000005e4: 9100c3fd    	add	x29, sp, #0x30
1000005e8: f9000be8    	str	x8, [sp, #0x10]
1000005ec: f81f83a8    	stur	x8, [x29, #-0x8]
1000005f0: f81f03a0    	stur	x0, [x29, #-0x10]
1000005f4: d10047a0    	sub	x0, x29, #0x11
1000005f8: f90007e0    	str	x0, [sp, #0x8]
1000005fc: 940000b0    	bl	0x1000008bc <__ZNSt3__19allocatorIiEC1B8ne200100Ev>
100000600: f94007e0    	ldr	x0, [sp, #0x8]
100000604: f9400be8    	ldr	x8, [sp, #0x10]
100000608: f85f03a1    	ldur	x1, [x29, #-0x10]
10000060c: 94000075    	bl	0x1000007e0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>
100000610: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000614: 910103ff    	add	sp, sp, #0x40
100000618: d65f03c0    	ret

000000010000061c <__ZNSt3__18weak_ptrIiEC1IiLi0EEERKNS_10shared_ptrIT_EE>:
10000061c: d100c3ff    	sub	sp, sp, #0x30
100000620: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000624: 910083fd    	add	x29, sp, #0x20
100000628: f81f83a0    	stur	x0, [x29, #-0x8]
10000062c: f9000be1    	str	x1, [sp, #0x10]
100000630: f85f83a0    	ldur	x0, [x29, #-0x8]
100000634: f90007e0    	str	x0, [sp, #0x8]
100000638: f9400be1    	ldr	x1, [sp, #0x10]
10000063c: 940003ae    	bl	0x1000014f4 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE>
100000640: f94007e0    	ldr	x0, [sp, #0x8]
100000644: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000648: 9100c3ff    	add	sp, sp, #0x30
10000064c: d65f03c0    	ret

0000000100000650 <__ZNKSt3__18weak_ptrIiE4lockEv>:
100000650: d10103ff    	sub	sp, sp, #0x40
100000654: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000658: 9100c3fd    	add	x29, sp, #0x30
10000065c: f90007e8    	str	x8, [sp, #0x8]
100000660: aa0003e8    	mov	x8, x0
100000664: f94007e0    	ldr	x0, [sp, #0x8]
100000668: aa0003e9    	mov	x9, x0
10000066c: f81f83a9    	stur	x9, [x29, #-0x8]
100000670: f81f03a8    	stur	x8, [x29, #-0x10]
100000674: f85f03a8    	ldur	x8, [x29, #-0x10]
100000678: f9000be8    	str	x8, [sp, #0x10]
10000067c: 52800008    	mov	w8, #0x0                ; =0
100000680: 12000108    	and	w8, w8, #0x1
100000684: 12000108    	and	w8, w8, #0x1
100000688: 381ef3a8    	sturb	w8, [x29, #-0x11]
10000068c: 94000307    	bl	0x1000012a8 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>
100000690: f9400be8    	ldr	x8, [sp, #0x10]
100000694: f9400508    	ldr	x8, [x8, #0x8]
100000698: b40000e8    	cbz	x8, 0x1000006b4 <__ZNKSt3__18weak_ptrIiE4lockEv+0x64>
10000069c: 14000001    	b	0x1000006a0 <__ZNKSt3__18weak_ptrIiE4lockEv+0x50>
1000006a0: f9400be8    	ldr	x8, [sp, #0x10]
1000006a4: f9400500    	ldr	x0, [x8, #0x8]
1000006a8: 940003dc    	bl	0x100001618 <___stack_chk_guard+0x100001618>
1000006ac: f90003e0    	str	x0, [sp]
1000006b0: 14000005    	b	0x1000006c4 <__ZNKSt3__18weak_ptrIiE4lockEv+0x74>
1000006b4: f9400be8    	ldr	x8, [sp, #0x10]
1000006b8: f9400508    	ldr	x8, [x8, #0x8]
1000006bc: f90003e8    	str	x8, [sp]
1000006c0: 14000001    	b	0x1000006c4 <__ZNKSt3__18weak_ptrIiE4lockEv+0x74>
1000006c4: f94007e8    	ldr	x8, [sp, #0x8]
1000006c8: f94003e9    	ldr	x9, [sp]
1000006cc: f9000509    	str	x9, [x8, #0x8]
1000006d0: f9400508    	ldr	x8, [x8, #0x8]
1000006d4: b40000e8    	cbz	x8, 0x1000006f0 <__ZNKSt3__18weak_ptrIiE4lockEv+0xa0>
1000006d8: 14000001    	b	0x1000006dc <__ZNKSt3__18weak_ptrIiE4lockEv+0x8c>
1000006dc: f94007e9    	ldr	x9, [sp, #0x8]
1000006e0: f9400be8    	ldr	x8, [sp, #0x10]
1000006e4: f9400108    	ldr	x8, [x8]
1000006e8: f9000128    	str	x8, [x9]
1000006ec: 14000001    	b	0x1000006f0 <__ZNKSt3__18weak_ptrIiE4lockEv+0xa0>
1000006f0: 52800028    	mov	w8, #0x1                ; =1
1000006f4: 12000108    	and	w8, w8, #0x1
1000006f8: 12000108    	and	w8, w8, #0x1
1000006fc: 381ef3a8    	sturb	w8, [x29, #-0x11]
100000700: 385ef3a8    	ldurb	w8, [x29, #-0x11]
100000704: 370000a8    	tbnz	w8, #0x0, 0x100000718 <__ZNKSt3__18weak_ptrIiE4lockEv+0xc8>
100000708: 14000001    	b	0x10000070c <__ZNKSt3__18weak_ptrIiE4lockEv+0xbc>
10000070c: f94007e0    	ldr	x0, [sp, #0x8]
100000710: 94000016    	bl	0x100000768 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000714: 14000001    	b	0x100000718 <__ZNKSt3__18weak_ptrIiE4lockEv+0xc8>
100000718: a9437bfd    	ldp	x29, x30, [sp, #0x30]
10000071c: 910103ff    	add	sp, sp, #0x40
100000720: d65f03c0    	ret

0000000100000724 <__ZNKSt3__110shared_ptrIiEcvbB8ne200100Ev>:
100000724: d10083ff    	sub	sp, sp, #0x20
100000728: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000072c: 910043fd    	add	x29, sp, #0x10
100000730: f90007e0    	str	x0, [sp, #0x8]
100000734: f94007e0    	ldr	x0, [sp, #0x8]
100000738: 940003b2    	bl	0x100001600 <__ZNKSt3__110shared_ptrIiE3getB8ne200100Ev>
10000073c: f1000008    	subs	x8, x0, #0x0
100000740: 1a9f07e0    	cset	w0, ne
100000744: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000748: 910083ff    	add	sp, sp, #0x20
10000074c: d65f03c0    	ret

0000000100000750 <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>:
100000750: d10043ff    	sub	sp, sp, #0x10
100000754: f90007e0    	str	x0, [sp, #0x8]
100000758: f94007e8    	ldr	x8, [sp, #0x8]
10000075c: f9400100    	ldr	x0, [x8]
100000760: 910043ff    	add	sp, sp, #0x10
100000764: d65f03c0    	ret

0000000100000768 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>:
100000768: d10083ff    	sub	sp, sp, #0x20
10000076c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000770: 910043fd    	add	x29, sp, #0x10
100000774: f90007e0    	str	x0, [sp, #0x8]
100000778: f94007e0    	ldr	x0, [sp, #0x8]
10000077c: f90003e0    	str	x0, [sp]
100000780: 94000311    	bl	0x1000013c4 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>
100000784: f94003e0    	ldr	x0, [sp]
100000788: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000078c: 910083ff    	add	sp, sp, #0x20
100000790: d65f03c0    	ret

0000000100000794 <__ZNSt3__18weak_ptrIiED1Ev>:
100000794: d10083ff    	sub	sp, sp, #0x20
100000798: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000079c: 910043fd    	add	x29, sp, #0x10
1000007a0: f90007e0    	str	x0, [sp, #0x8]
1000007a4: f94007e0    	ldr	x0, [sp, #0x8]
1000007a8: f90003e0    	str	x0, [sp]
1000007ac: 94000382    	bl	0x1000015b4 <__ZNSt3__18weak_ptrIiED2Ev>
1000007b0: f94003e0    	ldr	x0, [sp]
1000007b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000007b8: 910083ff    	add	sp, sp, #0x20
1000007bc: d65f03c0    	ret

00000001000007c0 <_main>:
1000007c0: d10083ff    	sub	sp, sp, #0x20
1000007c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000007c8: 910043fd    	add	x29, sp, #0x10
1000007cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000007d0: 97ffff5a    	bl	0x100000538 <__Z13test_weak_ptrv>
1000007d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000007d8: 910083ff    	add	sp, sp, #0x20
1000007dc: d65f03c0    	ret

00000001000007e0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>:
1000007e0: d10203ff    	sub	sp, sp, #0x80
1000007e4: a9077bfd    	stp	x29, x30, [sp, #0x70]
1000007e8: 9101c3fd    	add	x29, sp, #0x70
1000007ec: f9000be8    	str	x8, [sp, #0x10]
1000007f0: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000007f4: f9400929    	ldr	x9, [x9, #0x10]
1000007f8: f9400129    	ldr	x9, [x9]
1000007fc: f81f83a9    	stur	x9, [x29, #-0x8]
100000800: f81d83a8    	stur	x8, [x29, #-0x28]
100000804: f81d03a0    	stur	x0, [x29, #-0x30]
100000808: f9001fe1    	str	x1, [sp, #0x38]
10000080c: d10083a0    	sub	x0, x29, #0x20
100000810: d2800021    	mov	x1, #0x1                ; =1
100000814: 94000035    	bl	0x1000008e8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>
100000818: 14000001    	b	0x10000081c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x3c>
10000081c: d10083a0    	sub	x0, x29, #0x20
100000820: 9400003f    	bl	0x10000091c <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>
100000824: f9401fe1    	ldr	x1, [sp, #0x38]
100000828: 94000043    	bl	0x100000934 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>
10000082c: 14000001    	b	0x100000830 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x50>
100000830: d10083a0    	sub	x0, x29, #0x20
100000834: f90007e0    	str	x0, [sp, #0x8]
100000838: 9400004c    	bl	0x100000968 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>
10000083c: f9000fe0    	str	x0, [sp, #0x18]
100000840: f9400fe0    	ldr	x0, [sp, #0x18]
100000844: 9400007b    	bl	0x100000a30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000848: f9400be8    	ldr	x8, [sp, #0x10]
10000084c: f9400fe1    	ldr	x1, [sp, #0x18]
100000850: 94000375    	bl	0x100001624 <___stack_chk_guard+0x100001624>
100000854: f94007e0    	ldr	x0, [sp, #0x8]
100000858: 94000080    	bl	0x100000a58 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
10000085c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000860: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000864: f9400908    	ldr	x8, [x8, #0x10]
100000868: f9400108    	ldr	x8, [x8]
10000086c: eb090108    	subs	x8, x8, x9
100000870: 54000060    	b.eq	0x10000087c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x9c>
100000874: 14000001    	b	0x100000878 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x98>
100000878: 9400036e    	bl	0x100001630 <___stack_chk_guard+0x100001630>
10000087c: a9477bfd    	ldp	x29, x30, [sp, #0x70]
100000880: 910203ff    	add	sp, sp, #0x80
100000884: d65f03c0    	ret
100000888: f90017e0    	str	x0, [sp, #0x28]
10000088c: aa0103e8    	mov	x8, x1
100000890: b90027e8    	str	w8, [sp, #0x24]
100000894: d10083a0    	sub	x0, x29, #0x20
100000898: 94000070    	bl	0x100000a58 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
10000089c: 14000001    	b	0x1000008a0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xc0>
1000008a0: f94017e0    	ldr	x0, [sp, #0x28]
1000008a4: f90003e0    	str	x0, [sp]
1000008a8: 14000003    	b	0x1000008b4 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
1000008ac: f90003e0    	str	x0, [sp]
1000008b0: 14000001    	b	0x1000008b4 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
1000008b4: f94003e0    	ldr	x0, [sp]
1000008b8: 94000361    	bl	0x10000163c <___stack_chk_guard+0x10000163c>

00000001000008bc <__ZNSt3__19allocatorIiEC1B8ne200100Ev>:
1000008bc: d10083ff    	sub	sp, sp, #0x20
1000008c0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000008c4: 910043fd    	add	x29, sp, #0x10
1000008c8: f90007e0    	str	x0, [sp, #0x8]
1000008cc: f94007e0    	ldr	x0, [sp, #0x8]
1000008d0: f90003e0    	str	x0, [sp]
1000008d4: 940002ac    	bl	0x100001384 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>
1000008d8: f94003e0    	ldr	x0, [sp]
1000008dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000008e0: 910083ff    	add	sp, sp, #0x20
1000008e4: d65f03c0    	ret

00000001000008e8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>:
1000008e8: d100c3ff    	sub	sp, sp, #0x30
1000008ec: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000008f0: 910083fd    	add	x29, sp, #0x20
1000008f4: f9000be0    	str	x0, [sp, #0x10]
1000008f8: f90007e1    	str	x1, [sp, #0x8]
1000008fc: f9400be0    	ldr	x0, [sp, #0x10]
100000900: f90003e0    	str	x0, [sp]
100000904: f94007e1    	ldr	x1, [sp, #0x8]
100000908: 9400005f    	bl	0x100000a84 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>
10000090c: f94003e0    	ldr	x0, [sp]
100000910: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000914: 9100c3ff    	add	sp, sp, #0x30
100000918: d65f03c0    	ret

000000010000091c <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>:
10000091c: d10043ff    	sub	sp, sp, #0x10
100000920: f90007e0    	str	x0, [sp, #0x8]
100000924: f94007e8    	ldr	x8, [sp, #0x8]
100000928: f9400900    	ldr	x0, [x8, #0x10]
10000092c: 910043ff    	add	sp, sp, #0x10
100000930: d65f03c0    	ret

0000000100000934 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000934: d100c3ff    	sub	sp, sp, #0x30
100000938: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000093c: 910083fd    	add	x29, sp, #0x20
100000940: f9000be0    	str	x0, [sp, #0x10]
100000944: f90007e1    	str	x1, [sp, #0x8]
100000948: f9400be0    	ldr	x0, [sp, #0x10]
10000094c: f90003e0    	str	x0, [sp]
100000950: f94007e1    	ldr	x1, [sp, #0x8]
100000954: 940000f1    	bl	0x100000d18 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>
100000958: f94003e0    	ldr	x0, [sp]
10000095c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000960: 9100c3ff    	add	sp, sp, #0x30
100000964: d65f03c0    	ret

0000000100000968 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>:
100000968: d10043ff    	sub	sp, sp, #0x10
10000096c: f90007e0    	str	x0, [sp, #0x8]
100000970: f94007e8    	ldr	x8, [sp, #0x8]
100000974: f9400909    	ldr	x9, [x8, #0x10]
100000978: f90003e9    	str	x9, [sp]
10000097c: f900091f    	str	xzr, [x8, #0x10]
100000980: f94003e0    	ldr	x0, [sp]
100000984: 910043ff    	add	sp, sp, #0x10
100000988: d65f03c0    	ret

000000010000098c <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_>:
10000098c: d10143ff    	sub	sp, sp, #0x50
100000990: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000994: 910103fd    	add	x29, sp, #0x40
100000998: f9000fe8    	str	x8, [sp, #0x18]
10000099c: aa0003e8    	mov	x8, x0
1000009a0: f9400fe0    	ldr	x0, [sp, #0x18]
1000009a4: aa0003e9    	mov	x9, x0
1000009a8: f81f83a9    	stur	x9, [x29, #-0x8]
1000009ac: f81f03a8    	stur	x8, [x29, #-0x10]
1000009b0: f81e83a1    	stur	x1, [x29, #-0x18]
1000009b4: 52800008    	mov	w8, #0x0                ; =0
1000009b8: 52800029    	mov	w9, #0x1                ; =1
1000009bc: b90023e9    	str	w9, [sp, #0x20]
1000009c0: 12000108    	and	w8, w8, #0x1
1000009c4: 12000108    	and	w8, w8, #0x1
1000009c8: 381e73a8    	sturb	w8, [x29, #-0x19]
1000009cc: 94000237    	bl	0x1000012a8 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>
1000009d0: f9400fe0    	ldr	x0, [sp, #0x18]
1000009d4: f85f03a8    	ldur	x8, [x29, #-0x10]
1000009d8: f9000008    	str	x8, [x0]
1000009dc: f85e83a8    	ldur	x8, [x29, #-0x18]
1000009e0: f9000408    	str	x8, [x0, #0x8]
1000009e4: f940000a    	ldr	x10, [x0]
1000009e8: f9400008    	ldr	x8, [x0]
1000009ec: 910003e9    	mov	x9, sp
1000009f0: f900012a    	str	x10, [x9]
1000009f4: f9000528    	str	x8, [x9, #0x8]
1000009f8: 94000237    	bl	0x1000012d4 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>
1000009fc: b94023e9    	ldr	w9, [sp, #0x20]
100000a00: 12000128    	and	w8, w9, #0x1
100000a04: 0a090108    	and	w8, w8, w9
100000a08: 381e73a8    	sturb	w8, [x29, #-0x19]
100000a0c: 385e73a8    	ldurb	w8, [x29, #-0x19]
100000a10: 370000a8    	tbnz	w8, #0x0, 0x100000a24 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
100000a14: 14000001    	b	0x100000a18 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x8c>
100000a18: f9400fe0    	ldr	x0, [sp, #0x18]
100000a1c: 97ffff53    	bl	0x100000768 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000a20: 14000001    	b	0x100000a24 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
100000a24: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000a28: 910143ff    	add	sp, sp, #0x50
100000a2c: d65f03c0    	ret

0000000100000a30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>:
100000a30: d10083ff    	sub	sp, sp, #0x20
100000a34: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a38: 910043fd    	add	x29, sp, #0x10
100000a3c: f90007e0    	str	x0, [sp, #0x8]
100000a40: f94007e8    	ldr	x8, [sp, #0x8]
100000a44: 91006100    	add	x0, x8, #0x18
100000a48: 9400022e    	bl	0x100001300 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>
100000a4c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a50: 910083ff    	add	sp, sp, #0x20
100000a54: d65f03c0    	ret

0000000100000a58 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>:
100000a58: d10083ff    	sub	sp, sp, #0x20
100000a5c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a60: 910043fd    	add	x29, sp, #0x10
100000a64: f90007e0    	str	x0, [sp, #0x8]
100000a68: f94007e0    	ldr	x0, [sp, #0x8]
100000a6c: f90003e0    	str	x0, [sp]
100000a70: 94000229    	bl	0x100001314 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>
100000a74: f94003e0    	ldr	x0, [sp]
100000a78: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a7c: 910083ff    	add	sp, sp, #0x20
100000a80: d65f03c0    	ret

0000000100000a84 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>:
100000a84: d100c3ff    	sub	sp, sp, #0x30
100000a88: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a8c: 910083fd    	add	x29, sp, #0x20
100000a90: f9000be0    	str	x0, [sp, #0x10]
100000a94: f90007e1    	str	x1, [sp, #0x8]
100000a98: f9400be0    	ldr	x0, [sp, #0x10]
100000a9c: f90003e0    	str	x0, [sp]
100000aa0: d10007a1    	sub	x1, x29, #0x1
100000aa4: 9400000c    	bl	0x100000ad4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000aa8: f94003e0    	ldr	x0, [sp]
100000aac: f94007e8    	ldr	x8, [sp, #0x8]
100000ab0: f9000408    	str	x8, [x0, #0x8]
100000ab4: f9400401    	ldr	x1, [x0, #0x8]
100000ab8: 94000014    	bl	0x100000b08 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>
100000abc: aa0003e8    	mov	x8, x0
100000ac0: f94003e0    	ldr	x0, [sp]
100000ac4: f9000808    	str	x8, [x0, #0x10]
100000ac8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000acc: 9100c3ff    	add	sp, sp, #0x30
100000ad0: d65f03c0    	ret

0000000100000ad4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>:
100000ad4: d100c3ff    	sub	sp, sp, #0x30
100000ad8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000adc: 910083fd    	add	x29, sp, #0x20
100000ae0: f81f83a0    	stur	x0, [x29, #-0x8]
100000ae4: f9000be1    	str	x1, [sp, #0x10]
100000ae8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000aec: f90007e0    	str	x0, [sp, #0x8]
100000af0: f9400be1    	ldr	x1, [sp, #0x10]
100000af4: 94000010    	bl	0x100000b34 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>
100000af8: f94007e0    	ldr	x0, [sp, #0x8]
100000afc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000b00: 9100c3ff    	add	sp, sp, #0x30
100000b04: d65f03c0    	ret

0000000100000b08 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>:
100000b08: d10083ff    	sub	sp, sp, #0x20
100000b0c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000b10: 910043fd    	add	x29, sp, #0x10
100000b14: f90007e0    	str	x0, [sp, #0x8]
100000b18: f90003e1    	str	x1, [sp]
100000b1c: f94007e0    	ldr	x0, [sp, #0x8]
100000b20: f94003e1    	ldr	x1, [sp]
100000b24: 94000015    	bl	0x100000b78 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>
100000b28: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000b2c: 910083ff    	add	sp, sp, #0x20
100000b30: d65f03c0    	ret

0000000100000b34 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>:
100000b34: d100c3ff    	sub	sp, sp, #0x30
100000b38: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000b3c: 910083fd    	add	x29, sp, #0x20
100000b40: f81f83a0    	stur	x0, [x29, #-0x8]
100000b44: f9000be1    	str	x1, [sp, #0x10]
100000b48: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b4c: f90007e0    	str	x0, [sp, #0x8]
100000b50: 94000005    	bl	0x100000b64 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>
100000b54: f94007e0    	ldr	x0, [sp, #0x8]
100000b58: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000b5c: 9100c3ff    	add	sp, sp, #0x30
100000b60: d65f03c0    	ret

0000000100000b64 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>:
100000b64: d10043ff    	sub	sp, sp, #0x10
100000b68: f90007e0    	str	x0, [sp, #0x8]
100000b6c: f94007e0    	ldr	x0, [sp, #0x8]
100000b70: 910043ff    	add	sp, sp, #0x10
100000b74: d65f03c0    	ret

0000000100000b78 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>:
100000b78: d100c3ff    	sub	sp, sp, #0x30
100000b7c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000b80: 910083fd    	add	x29, sp, #0x20
100000b84: f81f83a0    	stur	x0, [x29, #-0x8]
100000b88: f9000be1    	str	x1, [sp, #0x10]
100000b8c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b90: f9400be8    	ldr	x8, [sp, #0x10]
100000b94: f90007e8    	str	x8, [sp, #0x8]
100000b98: 940002ac    	bl	0x100001648 <___stack_chk_guard+0x100001648>
100000b9c: f94007e8    	ldr	x8, [sp, #0x8]
100000ba0: eb000108    	subs	x8, x8, x0
100000ba4: 54000069    	b.ls	0x100000bb0 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x38>
100000ba8: 14000001    	b	0x100000bac <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x34>
100000bac: 94000011    	bl	0x100000bf0 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
100000bb0: f9400be0    	ldr	x0, [sp, #0x10]
100000bb4: d2800101    	mov	x1, #0x8                ; =8
100000bb8: 9400001b    	bl	0x100000c24 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>
100000bbc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000bc0: 9100c3ff    	add	sp, sp, #0x30
100000bc4: d65f03c0    	ret

0000000100000bc8 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8max_sizeB8ne200100IS5_vLi0EEEmRKS5_>:
100000bc8: d10083ff    	sub	sp, sp, #0x20
100000bcc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bd0: 910043fd    	add	x29, sp, #0x10
100000bd4: f90007e0    	str	x0, [sp, #0x8]
100000bd8: 9400002e    	bl	0x100000c90 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000bdc: d2800408    	mov	x8, #0x20               ; =32
100000be0: 9ac80800    	udiv	x0, x0, x8
100000be4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000be8: 910083ff    	add	sp, sp, #0x20
100000bec: d65f03c0    	ret

0000000100000bf0 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000bf0: d10083ff    	sub	sp, sp, #0x20
100000bf4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bf8: 910043fd    	add	x29, sp, #0x10
100000bfc: d2800100    	mov	x0, #0x8                ; =8
100000c00: 94000295    	bl	0x100001654 <___stack_chk_guard+0x100001654>
100000c04: f90007e0    	str	x0, [sp, #0x8]
100000c08: 94000296    	bl	0x100001660 <___stack_chk_guard+0x100001660>
100000c0c: f94007e0    	ldr	x0, [sp, #0x8]
100000c10: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000c14: f9402421    	ldr	x1, [x1, #0x48]
100000c18: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000c1c: f9402842    	ldr	x2, [x2, #0x50]
100000c20: 94000293    	bl	0x10000166c <___stack_chk_guard+0x10000166c>

0000000100000c24 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>:
100000c24: d10103ff    	sub	sp, sp, #0x40
100000c28: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000c2c: 9100c3fd    	add	x29, sp, #0x30
100000c30: f81f03a0    	stur	x0, [x29, #-0x10]
100000c34: f9000fe1    	str	x1, [sp, #0x18]
100000c38: f85f03a8    	ldur	x8, [x29, #-0x10]
100000c3c: d37be908    	lsl	x8, x8, #5
100000c40: f9000be8    	str	x8, [sp, #0x10]
100000c44: f9400fe0    	ldr	x0, [sp, #0x18]
100000c48: 94000019    	bl	0x100000cac <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000c4c: 36000120    	tbz	w0, #0x0, 0x100000c70 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x4c>
100000c50: 14000001    	b	0x100000c54 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x30>
100000c54: f9400fe8    	ldr	x8, [sp, #0x18]
100000c58: f90007e8    	str	x8, [sp, #0x8]
100000c5c: f9400be0    	ldr	x0, [sp, #0x10]
100000c60: f94007e1    	ldr	x1, [sp, #0x8]
100000c64: 94000019    	bl	0x100000cc8 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>
100000c68: f81f83a0    	stur	x0, [x29, #-0x8]
100000c6c: 14000005    	b	0x100000c80 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000c70: f9400be0    	ldr	x0, [sp, #0x10]
100000c74: 94000020    	bl	0x100000cf4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>
100000c78: f81f83a0    	stur	x0, [x29, #-0x8]
100000c7c: 14000001    	b	0x100000c80 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000c80: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c84: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000c88: 910103ff    	add	sp, sp, #0x40
100000c8c: d65f03c0    	ret

0000000100000c90 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000c90: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000c94: 910003fd    	mov	x29, sp
100000c98: 94000003    	bl	0x100000ca4 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000c9c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000ca0: d65f03c0    	ret

0000000100000ca4 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000ca4: 92800000    	mov	x0, #-0x1               ; =-1
100000ca8: d65f03c0    	ret

0000000100000cac <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000cac: d10043ff    	sub	sp, sp, #0x10
100000cb0: f90007e0    	str	x0, [sp, #0x8]
100000cb4: f94007e8    	ldr	x8, [sp, #0x8]
100000cb8: f1004108    	subs	x8, x8, #0x10
100000cbc: 1a9f97e0    	cset	w0, hi
100000cc0: 910043ff    	add	sp, sp, #0x10
100000cc4: d65f03c0    	ret

0000000100000cc8 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>:
100000cc8: d10083ff    	sub	sp, sp, #0x20
100000ccc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000cd0: 910043fd    	add	x29, sp, #0x10
100000cd4: f90007e0    	str	x0, [sp, #0x8]
100000cd8: f90003e1    	str	x1, [sp]
100000cdc: f94007e0    	ldr	x0, [sp, #0x8]
100000ce0: f94003e1    	ldr	x1, [sp]
100000ce4: 94000265    	bl	0x100001678 <___stack_chk_guard+0x100001678>
100000ce8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cec: 910083ff    	add	sp, sp, #0x20
100000cf0: d65f03c0    	ret

0000000100000cf4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>:
100000cf4: d10083ff    	sub	sp, sp, #0x20
100000cf8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000cfc: 910043fd    	add	x29, sp, #0x10
100000d00: f90007e0    	str	x0, [sp, #0x8]
100000d04: f94007e0    	ldr	x0, [sp, #0x8]
100000d08: 9400025f    	bl	0x100001684 <___stack_chk_guard+0x100001684>
100000d0c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d10: 910083ff    	add	sp, sp, #0x20
100000d14: d65f03c0    	ret

0000000100000d18 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000d18: d10103ff    	sub	sp, sp, #0x40
100000d1c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000d20: 9100c3fd    	add	x29, sp, #0x30
100000d24: f81f03a0    	stur	x0, [x29, #-0x10]
100000d28: f9000fe1    	str	x1, [sp, #0x18]
100000d2c: f85f03a0    	ldur	x0, [x29, #-0x10]
100000d30: f90003e0    	str	x0, [sp]
100000d34: d2800001    	mov	x1, #0x0                ; =0
100000d38: 94000027    	bl	0x100000dd4 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000d3c: f94003e8    	ldr	x8, [sp]
100000d40: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000d44: 91032129    	add	x9, x9, #0xc8
100000d48: 91004129    	add	x9, x9, #0x10
100000d4c: f9000109    	str	x9, [x8]
100000d50: 91006100    	add	x0, x8, #0x18
100000d54: d10007a1    	sub	x1, x29, #0x1
100000d58: 94000032    	bl	0x100000e20 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>
100000d5c: 14000001    	b	0x100000d60 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x48>
100000d60: f94003e0    	ldr	x0, [sp]
100000d64: 9400003c    	bl	0x100000e54 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000d68: f94003e0    	ldr	x0, [sp]
100000d6c: 97ffff31    	bl	0x100000a30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000d70: aa0003e1    	mov	x1, x0
100000d74: f9400fe2    	ldr	x2, [sp, #0x18]
100000d78: 91002fe0    	add	x0, sp, #0xb
100000d7c: 94000245    	bl	0x100001690 <___stack_chk_guard+0x100001690>
100000d80: 14000001    	b	0x100000d84 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x6c>
100000d84: f94003e0    	ldr	x0, [sp]
100000d88: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000d8c: 910103ff    	add	sp, sp, #0x40
100000d90: d65f03c0    	ret
100000d94: f9000be0    	str	x0, [sp, #0x10]
100000d98: aa0103e8    	mov	x8, x1
100000d9c: b9000fe8    	str	w8, [sp, #0xc]
100000da0: 14000008    	b	0x100000dc0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000da4: f94003e8    	ldr	x8, [sp]
100000da8: f9000be0    	str	x0, [sp, #0x10]
100000dac: aa0103e9    	mov	x9, x1
100000db0: b9000fe9    	str	w9, [sp, #0xc]
100000db4: 91006100    	add	x0, x8, #0x18
100000db8: 9400003d    	bl	0x100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000dbc: 14000001    	b	0x100000dc0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000dc0: f94003e0    	ldr	x0, [sp]
100000dc4: 94000236    	bl	0x10000169c <___stack_chk_guard+0x10000169c>
100000dc8: 14000001    	b	0x100000dcc <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xb4>
100000dcc: f9400be0    	ldr	x0, [sp, #0x10]
100000dd0: 9400021b    	bl	0x10000163c <___stack_chk_guard+0x10000163c>

0000000100000dd4 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000dd4: d100c3ff    	sub	sp, sp, #0x30
100000dd8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000ddc: 910083fd    	add	x29, sp, #0x20
100000de0: f81f83a0    	stur	x0, [x29, #-0x8]
100000de4: f9000be1    	str	x1, [sp, #0x10]
100000de8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000dec: f90007e0    	str	x0, [sp, #0x8]
100000df0: f9400be1    	ldr	x1, [sp, #0x10]
100000df4: 94000070    	bl	0x100000fb4 <__ZNSt3__114__shared_countC2B8ne200100El>
100000df8: f94007e0    	ldr	x0, [sp, #0x8]
100000dfc: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000e00: f9404108    	ldr	x8, [x8, #0x80]
100000e04: 91004108    	add	x8, x8, #0x10
100000e08: f9000008    	str	x8, [x0]
100000e0c: f9400be8    	ldr	x8, [sp, #0x10]
100000e10: f9000808    	str	x8, [x0, #0x10]
100000e14: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e18: 9100c3ff    	add	sp, sp, #0x30
100000e1c: d65f03c0    	ret

0000000100000e20 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>:
100000e20: d100c3ff    	sub	sp, sp, #0x30
100000e24: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e28: 910083fd    	add	x29, sp, #0x20
100000e2c: f81f83a0    	stur	x0, [x29, #-0x8]
100000e30: f9000be1    	str	x1, [sp, #0x10]
100000e34: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e38: f90007e0    	str	x0, [sp, #0x8]
100000e3c: f9400be1    	ldr	x1, [sp, #0x10]
100000e40: 94000069    	bl	0x100000fe4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>
100000e44: f94007e0    	ldr	x0, [sp, #0x8]
100000e48: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e4c: 9100c3ff    	add	sp, sp, #0x30
100000e50: d65f03c0    	ret

0000000100000e54 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>:
100000e54: d10083ff    	sub	sp, sp, #0x20
100000e58: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e5c: 910043fd    	add	x29, sp, #0x10
100000e60: f90007e0    	str	x0, [sp, #0x8]
100000e64: f94007e8    	ldr	x8, [sp, #0x8]
100000e68: 91006100    	add	x0, x8, #0x18
100000e6c: 9400006a    	bl	0x100001014 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000e70: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e74: 910083ff    	add	sp, sp, #0x20
100000e78: d65f03c0    	ret

0000000100000e7c <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE9constructB8ne200100IiJiEvLi0EEEvRS2_PT_DpOT0_>:
100000e7c: d100c3ff    	sub	sp, sp, #0x30
100000e80: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e84: 910083fd    	add	x29, sp, #0x20
100000e88: f81f83a0    	stur	x0, [x29, #-0x8]
100000e8c: f9000be1    	str	x1, [sp, #0x10]
100000e90: f90007e2    	str	x2, [sp, #0x8]
100000e94: f9400be0    	ldr	x0, [sp, #0x10]
100000e98: f94007e1    	ldr	x1, [sp, #0x8]
100000e9c: 94000063    	bl	0x100001028 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000ea0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ea4: 9100c3ff    	add	sp, sp, #0x30
100000ea8: d65f03c0    	ret

0000000100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>:
100000eac: d10083ff    	sub	sp, sp, #0x20
100000eb0: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000eb4: 910043fd    	add	x29, sp, #0x10
100000eb8: f90007e0    	str	x0, [sp, #0x8]
100000ebc: f94007e0    	ldr	x0, [sp, #0x8]
100000ec0: f90003e0    	str	x0, [sp]
100000ec4: 9400006d    	bl	0x100001078 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>
100000ec8: f94003e0    	ldr	x0, [sp]
100000ecc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ed0: 910083ff    	add	sp, sp, #0x20
100000ed4: d65f03c0    	ret

0000000100000ed8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000ed8: d10083ff    	sub	sp, sp, #0x20
100000edc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ee0: 910043fd    	add	x29, sp, #0x10
100000ee4: f90007e0    	str	x0, [sp, #0x8]
100000ee8: f94007e0    	ldr	x0, [sp, #0x8]
100000eec: f90003e0    	str	x0, [sp]
100000ef0: 9400006d    	bl	0x1000010a4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>
100000ef4: f94003e0    	ldr	x0, [sp]
100000ef8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000efc: 910083ff    	add	sp, sp, #0x20
100000f00: d65f03c0    	ret

0000000100000f04 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000f04: d10083ff    	sub	sp, sp, #0x20
100000f08: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f0c: 910043fd    	add	x29, sp, #0x10
100000f10: f90007e0    	str	x0, [sp, #0x8]
100000f14: f94007e0    	ldr	x0, [sp, #0x8]
100000f18: f90003e0    	str	x0, [sp]
100000f1c: 97ffffef    	bl	0x100000ed8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>
100000f20: f94003e0    	ldr	x0, [sp]
100000f24: 940001e1    	bl	0x1000016a8 <___stack_chk_guard+0x1000016a8>
100000f28: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f2c: 910083ff    	add	sp, sp, #0x20
100000f30: d65f03c0    	ret

0000000100000f34 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000f34: d10083ff    	sub	sp, sp, #0x20
100000f38: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f3c: 910043fd    	add	x29, sp, #0x10
100000f40: f90007e0    	str	x0, [sp, #0x8]
100000f44: f94007e0    	ldr	x0, [sp, #0x8]
100000f48: 940001db    	bl	0x1000016b4 <___stack_chk_guard+0x1000016b4>
100000f4c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f50: 910083ff    	add	sp, sp, #0x20
100000f54: d65f03c0    	ret

0000000100000f58 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000f58: d100c3ff    	sub	sp, sp, #0x30
100000f5c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000f60: 910083fd    	add	x29, sp, #0x20
100000f64: f81f83a0    	stur	x0, [x29, #-0x8]
100000f68: f85f83a0    	ldur	x0, [x29, #-0x8]
100000f6c: f90003e0    	str	x0, [sp]
100000f70: 97ffffb9    	bl	0x100000e54 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000f74: aa0003e1    	mov	x1, x0
100000f78: d10027a0    	sub	x0, x29, #0x9
100000f7c: f90007e0    	str	x0, [sp, #0x8]
100000f80: 97fffed5    	bl	0x100000ad4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000f84: f94003e8    	ldr	x8, [sp]
100000f88: 91006100    	add	x0, x8, #0x18
100000f8c: 97ffffc8    	bl	0x100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000f90: f94003e0    	ldr	x0, [sp]
100000f94: 94000086    	bl	0x1000011ac <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>
100000f98: aa0003e1    	mov	x1, x0
100000f9c: f94007e0    	ldr	x0, [sp, #0x8]
100000fa0: d2800022    	mov	x2, #0x1                ; =1
100000fa4: 94000075    	bl	0x100001178 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100000fa8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000fac: 9100c3ff    	add	sp, sp, #0x30
100000fb0: d65f03c0    	ret

0000000100000fb4 <__ZNSt3__114__shared_countC2B8ne200100El>:
100000fb4: d10043ff    	sub	sp, sp, #0x10
100000fb8: f90007e0    	str	x0, [sp, #0x8]
100000fbc: f90003e1    	str	x1, [sp]
100000fc0: f94007e0    	ldr	x0, [sp, #0x8]
100000fc4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000fc8: f9404d08    	ldr	x8, [x8, #0x98]
100000fcc: 91004108    	add	x8, x8, #0x10
100000fd0: f9000008    	str	x8, [x0]
100000fd4: f94003e8    	ldr	x8, [sp]
100000fd8: f9000408    	str	x8, [x0, #0x8]
100000fdc: 910043ff    	add	sp, sp, #0x10
100000fe0: d65f03c0    	ret

0000000100000fe4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>:
100000fe4: d100c3ff    	sub	sp, sp, #0x30
100000fe8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000fec: 910083fd    	add	x29, sp, #0x20
100000ff0: f81f83a0    	stur	x0, [x29, #-0x8]
100000ff4: f9000be1    	str	x1, [sp, #0x10]
100000ff8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000ffc: f90007e0    	str	x0, [sp, #0x8]
100001000: 94000005    	bl	0x100001014 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100001004: f94007e0    	ldr	x0, [sp, #0x8]
100001008: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000100c: 9100c3ff    	add	sp, sp, #0x30
100001010: d65f03c0    	ret

0000000100001014 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>:
100001014: d10043ff    	sub	sp, sp, #0x10
100001018: f90007e0    	str	x0, [sp, #0x8]
10000101c: f94007e0    	ldr	x0, [sp, #0x8]
100001020: 910043ff    	add	sp, sp, #0x10
100001024: d65f03c0    	ret

0000000100001028 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100001028: d10083ff    	sub	sp, sp, #0x20
10000102c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001030: 910043fd    	add	x29, sp, #0x10
100001034: f90007e0    	str	x0, [sp, #0x8]
100001038: f90003e1    	str	x1, [sp]
10000103c: f94007e0    	ldr	x0, [sp, #0x8]
100001040: f94003e1    	ldr	x1, [sp]
100001044: 94000004    	bl	0x100001054 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100001048: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000104c: 910083ff    	add	sp, sp, #0x20
100001050: d65f03c0    	ret

0000000100001054 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100001054: d10043ff    	sub	sp, sp, #0x10
100001058: f90007e0    	str	x0, [sp, #0x8]
10000105c: f90003e1    	str	x1, [sp]
100001060: f94007e0    	ldr	x0, [sp, #0x8]
100001064: f94003e8    	ldr	x8, [sp]
100001068: b9400108    	ldr	w8, [x8]
10000106c: b9000008    	str	w8, [x0]
100001070: 910043ff    	add	sp, sp, #0x10
100001074: d65f03c0    	ret

0000000100001078 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>:
100001078: d10083ff    	sub	sp, sp, #0x20
10000107c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001080: 910043fd    	add	x29, sp, #0x10
100001084: f90007e0    	str	x0, [sp, #0x8]
100001088: f94007e0    	ldr	x0, [sp, #0x8]
10000108c: f90003e0    	str	x0, [sp]
100001090: 97ffffe1    	bl	0x100001014 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100001094: f94003e0    	ldr	x0, [sp]
100001098: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000109c: 910083ff    	add	sp, sp, #0x20
1000010a0: d65f03c0    	ret

00000001000010a4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>:
1000010a4: d10083ff    	sub	sp, sp, #0x20
1000010a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000010ac: 910043fd    	add	x29, sp, #0x10
1000010b0: f90007e0    	str	x0, [sp, #0x8]
1000010b4: f94007e8    	ldr	x8, [sp, #0x8]
1000010b8: f90003e8    	str	x8, [sp]
1000010bc: f0000009    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000010c0: 91032129    	add	x9, x9, #0xc8
1000010c4: 91004129    	add	x9, x9, #0x10
1000010c8: f9000109    	str	x9, [x8]
1000010cc: 91006100    	add	x0, x8, #0x18
1000010d0: 97ffff77    	bl	0x100000eac <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
1000010d4: f94003e0    	ldr	x0, [sp]
1000010d8: 94000171    	bl	0x10000169c <___stack_chk_guard+0x10000169c>
1000010dc: f94003e0    	ldr	x0, [sp]
1000010e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000010e4: 910083ff    	add	sp, sp, #0x20
1000010e8: d65f03c0    	ret

00000001000010ec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv>:
1000010ec: d100c3ff    	sub	sp, sp, #0x30
1000010f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000010f4: 910083fd    	add	x29, sp, #0x20
1000010f8: f81f83a0    	stur	x0, [x29, #-0x8]
1000010fc: f85f83a0    	ldur	x0, [x29, #-0x8]
100001100: f90007e0    	str	x0, [sp, #0x8]
100001104: 97ffff54    	bl	0x100000e54 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100001108: f94007e0    	ldr	x0, [sp, #0x8]
10000110c: 97fffe49    	bl	0x100000a30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100001110: aa0003e1    	mov	x1, x0
100001114: d10027a0    	sub	x0, x29, #0x9
100001118: 9400016a    	bl	0x1000016c0 <___stack_chk_guard+0x1000016c0>
10000111c: 14000001    	b	0x100001120 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv+0x34>
100001120: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001124: 9100c3ff    	add	sp, sp, #0x30
100001128: d65f03c0    	ret
10000112c: 9400000b    	bl	0x100001158 <___clang_call_terminate>

0000000100001130 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE7destroyB8ne200100IivLi0EEEvRS2_PT_>:
100001130: d10083ff    	sub	sp, sp, #0x20
100001134: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001138: 910043fd    	add	x29, sp, #0x10
10000113c: f90007e0    	str	x0, [sp, #0x8]
100001140: f90003e1    	str	x1, [sp]
100001144: f94003e0    	ldr	x0, [sp]
100001148: 94000008    	bl	0x100001168 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>
10000114c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001150: 910083ff    	add	sp, sp, #0x20
100001154: d65f03c0    	ret

0000000100001158 <___clang_call_terminate>:
100001158: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000115c: 910003fd    	mov	x29, sp
100001160: 9400015b    	bl	0x1000016cc <___stack_chk_guard+0x1000016cc>
100001164: 9400015d    	bl	0x1000016d8 <___stack_chk_guard+0x1000016d8>

0000000100001168 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>:
100001168: d10043ff    	sub	sp, sp, #0x10
10000116c: f90007e0    	str	x0, [sp, #0x8]
100001170: 910043ff    	add	sp, sp, #0x10
100001174: d65f03c0    	ret

0000000100001178 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>:
100001178: d100c3ff    	sub	sp, sp, #0x30
10000117c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001180: 910083fd    	add	x29, sp, #0x20
100001184: f81f83a0    	stur	x0, [x29, #-0x8]
100001188: f9000be1    	str	x1, [sp, #0x10]
10000118c: f90007e2    	str	x2, [sp, #0x8]
100001190: f85f83a0    	ldur	x0, [x29, #-0x8]
100001194: f9400be1    	ldr	x1, [sp, #0x10]
100001198: f94007e2    	ldr	x2, [sp, #0x8]
10000119c: 94000009    	bl	0x1000011c0 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>
1000011a0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000011a4: 9100c3ff    	add	sp, sp, #0x30
1000011a8: d65f03c0    	ret

00000001000011ac <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>:
1000011ac: d10043ff    	sub	sp, sp, #0x10
1000011b0: f90007e0    	str	x0, [sp, #0x8]
1000011b4: f94007e0    	ldr	x0, [sp, #0x8]
1000011b8: 910043ff    	add	sp, sp, #0x10
1000011bc: d65f03c0    	ret

00000001000011c0 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>:
1000011c0: d100c3ff    	sub	sp, sp, #0x30
1000011c4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000011c8: 910083fd    	add	x29, sp, #0x20
1000011cc: f81f83a0    	stur	x0, [x29, #-0x8]
1000011d0: f9000be1    	str	x1, [sp, #0x10]
1000011d4: f90007e2    	str	x2, [sp, #0x8]
1000011d8: f9400be0    	ldr	x0, [sp, #0x10]
1000011dc: f94007e1    	ldr	x1, [sp, #0x8]
1000011e0: d2800102    	mov	x2, #0x8                ; =8
1000011e4: 94000004    	bl	0x1000011f4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
1000011e8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000011ec: 9100c3ff    	add	sp, sp, #0x30
1000011f0: d65f03c0    	ret

00000001000011f4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
1000011f4: d10103ff    	sub	sp, sp, #0x40
1000011f8: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000011fc: 9100c3fd    	add	x29, sp, #0x30
100001200: f81f83a0    	stur	x0, [x29, #-0x8]
100001204: f81f03a1    	stur	x1, [x29, #-0x10]
100001208: f9000fe2    	str	x2, [sp, #0x18]
10000120c: f85f03a8    	ldur	x8, [x29, #-0x10]
100001210: d37be908    	lsl	x8, x8, #5
100001214: f9000be8    	str	x8, [sp, #0x10]
100001218: f9400fe0    	ldr	x0, [sp, #0x18]
10000121c: 97fffea4    	bl	0x100000cac <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100001220: 36000100    	tbz	w0, #0x0, 0x100001240 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x4c>
100001224: 14000001    	b	0x100001228 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x34>
100001228: f9400fe8    	ldr	x8, [sp, #0x18]
10000122c: f90007e8    	str	x8, [sp, #0x8]
100001230: f85f83a0    	ldur	x0, [x29, #-0x8]
100001234: f94007e1    	ldr	x1, [sp, #0x8]
100001238: 94000008    	bl	0x100001258 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>
10000123c: 14000004    	b	0x10000124c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
100001240: f85f83a0    	ldur	x0, [x29, #-0x8]
100001244: 94000010    	bl	0x100001284 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>
100001248: 14000001    	b	0x10000124c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
10000124c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100001250: 910103ff    	add	sp, sp, #0x40
100001254: d65f03c0    	ret

0000000100001258 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>:
100001258: d10083ff    	sub	sp, sp, #0x20
10000125c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001260: 910043fd    	add	x29, sp, #0x10
100001264: f90007e0    	str	x0, [sp, #0x8]
100001268: f90003e1    	str	x1, [sp]
10000126c: f94007e0    	ldr	x0, [sp, #0x8]
100001270: f94003e1    	ldr	x1, [sp]
100001274: 9400011c    	bl	0x1000016e4 <___stack_chk_guard+0x1000016e4>
100001278: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000127c: 910083ff    	add	sp, sp, #0x20
100001280: d65f03c0    	ret

0000000100001284 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>:
100001284: d10083ff    	sub	sp, sp, #0x20
100001288: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000128c: 910043fd    	add	x29, sp, #0x10
100001290: f90007e0    	str	x0, [sp, #0x8]
100001294: f94007e0    	ldr	x0, [sp, #0x8]
100001298: 94000104    	bl	0x1000016a8 <___stack_chk_guard+0x1000016a8>
10000129c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012a0: 910083ff    	add	sp, sp, #0x20
1000012a4: d65f03c0    	ret

00000001000012a8 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>:
1000012a8: d10083ff    	sub	sp, sp, #0x20
1000012ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012b0: 910043fd    	add	x29, sp, #0x10
1000012b4: f90007e0    	str	x0, [sp, #0x8]
1000012b8: f94007e0    	ldr	x0, [sp, #0x8]
1000012bc: f90003e0    	str	x0, [sp]
1000012c0: 94000009    	bl	0x1000012e4 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>
1000012c4: f94003e0    	ldr	x0, [sp]
1000012c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012cc: 910083ff    	add	sp, sp, #0x20
1000012d0: d65f03c0    	ret

00000001000012d4 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>:
1000012d4: d10043ff    	sub	sp, sp, #0x10
1000012d8: f90007e0    	str	x0, [sp, #0x8]
1000012dc: 910043ff    	add	sp, sp, #0x10
1000012e0: d65f03c0    	ret

00000001000012e4 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>:
1000012e4: d10043ff    	sub	sp, sp, #0x10
1000012e8: f90007e0    	str	x0, [sp, #0x8]
1000012ec: f94007e0    	ldr	x0, [sp, #0x8]
1000012f0: f900001f    	str	xzr, [x0]
1000012f4: f900041f    	str	xzr, [x0, #0x8]
1000012f8: 910043ff    	add	sp, sp, #0x10
1000012fc: d65f03c0    	ret

0000000100001300 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>:
100001300: d10043ff    	sub	sp, sp, #0x10
100001304: f90007e0    	str	x0, [sp, #0x8]
100001308: f94007e0    	ldr	x0, [sp, #0x8]
10000130c: 910043ff    	add	sp, sp, #0x10
100001310: d65f03c0    	ret

0000000100001314 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>:
100001314: d10083ff    	sub	sp, sp, #0x20
100001318: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000131c: 910043fd    	add	x29, sp, #0x10
100001320: f90007e0    	str	x0, [sp, #0x8]
100001324: f94007e0    	ldr	x0, [sp, #0x8]
100001328: f90003e0    	str	x0, [sp]
10000132c: 94000005    	bl	0x100001340 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>
100001330: f94003e0    	ldr	x0, [sp]
100001334: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001338: 910083ff    	add	sp, sp, #0x20
10000133c: d65f03c0    	ret

0000000100001340 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>:
100001340: d10083ff    	sub	sp, sp, #0x20
100001344: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001348: 910043fd    	add	x29, sp, #0x10
10000134c: f90007e0    	str	x0, [sp, #0x8]
100001350: f94007e8    	ldr	x8, [sp, #0x8]
100001354: f90003e8    	str	x8, [sp]
100001358: f9400908    	ldr	x8, [x8, #0x10]
10000135c: b40000e8    	cbz	x8, 0x100001378 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
100001360: 14000001    	b	0x100001364 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x24>
100001364: f94003e0    	ldr	x0, [sp]
100001368: f9400801    	ldr	x1, [x0, #0x10]
10000136c: f9400402    	ldr	x2, [x0, #0x8]
100001370: 97ffff82    	bl	0x100001178 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100001374: 14000001    	b	0x100001378 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
100001378: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000137c: 910083ff    	add	sp, sp, #0x20
100001380: d65f03c0    	ret

0000000100001384 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>:
100001384: d10083ff    	sub	sp, sp, #0x20
100001388: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000138c: 910043fd    	add	x29, sp, #0x10
100001390: f90007e0    	str	x0, [sp, #0x8]
100001394: f94007e0    	ldr	x0, [sp, #0x8]
100001398: f90003e0    	str	x0, [sp]
10000139c: 94000005    	bl	0x1000013b0 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
1000013a0: f94003e0    	ldr	x0, [sp]
1000013a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000013a8: 910083ff    	add	sp, sp, #0x20
1000013ac: d65f03c0    	ret

00000001000013b0 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
1000013b0: d10043ff    	sub	sp, sp, #0x10
1000013b4: f90007e0    	str	x0, [sp, #0x8]
1000013b8: f94007e0    	ldr	x0, [sp, #0x8]
1000013bc: 910043ff    	add	sp, sp, #0x10
1000013c0: d65f03c0    	ret

00000001000013c4 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>:
1000013c4: d100c3ff    	sub	sp, sp, #0x30
1000013c8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000013cc: 910083fd    	add	x29, sp, #0x20
1000013d0: f9000be0    	str	x0, [sp, #0x10]
1000013d4: f9400be8    	ldr	x8, [sp, #0x10]
1000013d8: f90007e8    	str	x8, [sp, #0x8]
1000013dc: aa0803e9    	mov	x9, x8
1000013e0: f81f83a9    	stur	x9, [x29, #-0x8]
1000013e4: f9400508    	ldr	x8, [x8, #0x8]
1000013e8: b40000c8    	cbz	x8, 0x100001400 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
1000013ec: 14000001    	b	0x1000013f0 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x2c>
1000013f0: f94007e8    	ldr	x8, [sp, #0x8]
1000013f4: f9400500    	ldr	x0, [x8, #0x8]
1000013f8: 94000006    	bl	0x100001410 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
1000013fc: 14000001    	b	0x100001400 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
100001400: f85f83a0    	ldur	x0, [x29, #-0x8]
100001404: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001408: 9100c3ff    	add	sp, sp, #0x30
10000140c: d65f03c0    	ret

0000000100001410 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
100001410: d10083ff    	sub	sp, sp, #0x20
100001414: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001418: 910043fd    	add	x29, sp, #0x10
10000141c: f90007e0    	str	x0, [sp, #0x8]
100001420: f94007e0    	ldr	x0, [sp, #0x8]
100001424: f90003e0    	str	x0, [sp]
100001428: 94000009    	bl	0x10000144c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
10000142c: 360000a0    	tbz	w0, #0x0, 0x100001440 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
100001430: 14000001    	b	0x100001434 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
100001434: f94003e0    	ldr	x0, [sp]
100001438: 940000ae    	bl	0x1000016f0 <___stack_chk_guard+0x1000016f0>
10000143c: 14000001    	b	0x100001440 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
100001440: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001444: 910083ff    	add	sp, sp, #0x20
100001448: d65f03c0    	ret

000000010000144c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
10000144c: d100c3ff    	sub	sp, sp, #0x30
100001450: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001454: 910083fd    	add	x29, sp, #0x20
100001458: f9000be0    	str	x0, [sp, #0x10]
10000145c: f9400be8    	ldr	x8, [sp, #0x10]
100001460: f90007e8    	str	x8, [sp, #0x8]
100001464: 91002100    	add	x0, x8, #0x8
100001468: 94000017    	bl	0x1000014c4 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
10000146c: b1000408    	adds	x8, x0, #0x1
100001470: 54000161    	b.ne	0x10000149c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
100001474: 14000001    	b	0x100001478 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
100001478: f94007e0    	ldr	x0, [sp, #0x8]
10000147c: f9400008    	ldr	x8, [x0]
100001480: f9400908    	ldr	x8, [x8, #0x10]
100001484: d63f0100    	blr	x8
100001488: 52800028    	mov	w8, #0x1                ; =1
10000148c: 12000108    	and	w8, w8, #0x1
100001490: 12000108    	and	w8, w8, #0x1
100001494: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001498: 14000006    	b	0x1000014b0 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
10000149c: 52800008    	mov	w8, #0x0                ; =0
1000014a0: 12000108    	and	w8, w8, #0x1
1000014a4: 12000108    	and	w8, w8, #0x1
1000014a8: 381ff3a8    	sturb	w8, [x29, #-0x1]
1000014ac: 14000001    	b	0x1000014b0 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
1000014b0: 385ff3a8    	ldurb	w8, [x29, #-0x1]
1000014b4: 12000100    	and	w0, w8, #0x1
1000014b8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000014bc: 9100c3ff    	add	sp, sp, #0x30
1000014c0: d65f03c0    	ret

00000001000014c4 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
1000014c4: d10083ff    	sub	sp, sp, #0x20
1000014c8: f9000fe0    	str	x0, [sp, #0x18]
1000014cc: f9400fe8    	ldr	x8, [sp, #0x18]
1000014d0: 92800009    	mov	x9, #-0x1               ; =-1
1000014d4: f9000be9    	str	x9, [sp, #0x10]
1000014d8: f9400be9    	ldr	x9, [sp, #0x10]
1000014dc: f8e90108    	ldaddal	x9, x8, [x8]
1000014e0: 8b090108    	add	x8, x8, x9
1000014e4: f90007e8    	str	x8, [sp, #0x8]
1000014e8: f94007e0    	ldr	x0, [sp, #0x8]
1000014ec: 910083ff    	add	sp, sp, #0x20
1000014f0: d65f03c0    	ret

00000001000014f4 <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE>:
1000014f4: d100c3ff    	sub	sp, sp, #0x30
1000014f8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000014fc: 910083fd    	add	x29, sp, #0x20
100001500: f9000be0    	str	x0, [sp, #0x10]
100001504: f90007e1    	str	x1, [sp, #0x8]
100001508: f9400be8    	ldr	x8, [sp, #0x10]
10000150c: f90003e8    	str	x8, [sp]
100001510: aa0803e9    	mov	x9, x8
100001514: f81f83a9    	stur	x9, [x29, #-0x8]
100001518: f94007e9    	ldr	x9, [sp, #0x8]
10000151c: f9400129    	ldr	x9, [x9]
100001520: f9000109    	str	x9, [x8]
100001524: f94007e9    	ldr	x9, [sp, #0x8]
100001528: f9400529    	ldr	x9, [x9, #0x8]
10000152c: f9000509    	str	x9, [x8, #0x8]
100001530: f9400508    	ldr	x8, [x8, #0x8]
100001534: b40000c8    	cbz	x8, 0x10000154c <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE+0x58>
100001538: 14000001    	b	0x10000153c <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE+0x48>
10000153c: f94003e8    	ldr	x8, [sp]
100001540: f9400500    	ldr	x0, [x8, #0x8]
100001544: 94000006    	bl	0x10000155c <__ZNSt3__119__shared_weak_count10__add_weakB8ne200100Ev>
100001548: 14000001    	b	0x10000154c <__ZNSt3__18weak_ptrIiEC2IiLi0EEERKNS_10shared_ptrIT_EE+0x58>
10000154c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001550: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001554: 9100c3ff    	add	sp, sp, #0x30
100001558: d65f03c0    	ret

000000010000155c <__ZNSt3__119__shared_weak_count10__add_weakB8ne200100Ev>:
10000155c: d10083ff    	sub	sp, sp, #0x20
100001560: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001564: 910043fd    	add	x29, sp, #0x10
100001568: f90007e0    	str	x0, [sp, #0x8]
10000156c: f94007e8    	ldr	x8, [sp, #0x8]
100001570: 91004100    	add	x0, x8, #0x10
100001574: 94000004    	bl	0x100001584 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>
100001578: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000157c: 910083ff    	add	sp, sp, #0x20
100001580: d65f03c0    	ret

0000000100001584 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>:
100001584: d10083ff    	sub	sp, sp, #0x20
100001588: f9000fe0    	str	x0, [sp, #0x18]
10000158c: f9400fe8    	ldr	x8, [sp, #0x18]
100001590: d2800029    	mov	x9, #0x1                ; =1
100001594: f9000be9    	str	x9, [sp, #0x10]
100001598: f9400be9    	ldr	x9, [sp, #0x10]
10000159c: f8290108    	ldadd	x9, x8, [x8]
1000015a0: 8b090108    	add	x8, x8, x9
1000015a4: f90007e8    	str	x8, [sp, #0x8]
1000015a8: f94007e0    	ldr	x0, [sp, #0x8]
1000015ac: 910083ff    	add	sp, sp, #0x20
1000015b0: d65f03c0    	ret

00000001000015b4 <__ZNSt3__18weak_ptrIiED2Ev>:
1000015b4: d100c3ff    	sub	sp, sp, #0x30
1000015b8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000015bc: 910083fd    	add	x29, sp, #0x20
1000015c0: f9000be0    	str	x0, [sp, #0x10]
1000015c4: f9400be8    	ldr	x8, [sp, #0x10]
1000015c8: f90007e8    	str	x8, [sp, #0x8]
1000015cc: aa0803e9    	mov	x9, x8
1000015d0: f81f83a9    	stur	x9, [x29, #-0x8]
1000015d4: f9400508    	ldr	x8, [x8, #0x8]
1000015d8: b40000c8    	cbz	x8, 0x1000015f0 <__ZNSt3__18weak_ptrIiED2Ev+0x3c>
1000015dc: 14000001    	b	0x1000015e0 <__ZNSt3__18weak_ptrIiED2Ev+0x2c>
1000015e0: f94007e8    	ldr	x8, [sp, #0x8]
1000015e4: f9400500    	ldr	x0, [x8, #0x8]
1000015e8: 94000042    	bl	0x1000016f0 <___stack_chk_guard+0x1000016f0>
1000015ec: 14000001    	b	0x1000015f0 <__ZNSt3__18weak_ptrIiED2Ev+0x3c>
1000015f0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000015f4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000015f8: 9100c3ff    	add	sp, sp, #0x30
1000015fc: d65f03c0    	ret

0000000100001600 <__ZNKSt3__110shared_ptrIiE3getB8ne200100Ev>:
100001600: d10043ff    	sub	sp, sp, #0x10
100001604: f90007e0    	str	x0, [sp, #0x8]
100001608: f94007e8    	ldr	x8, [sp, #0x8]
10000160c: f9400100    	ldr	x0, [x8]
100001610: 910043ff    	add	sp, sp, #0x10
100001614: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001618 <__stubs>:
100001618: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000161c: f9400610    	ldr	x16, [x16, #0x8]
100001620: d61f0200    	br	x16
100001624: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001628: f9400e10    	ldr	x16, [x16, #0x18]
10000162c: d61f0200    	br	x16
100001630: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001634: f9401210    	ldr	x16, [x16, #0x20]
100001638: d61f0200    	br	x16
10000163c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001640: f9401610    	ldr	x16, [x16, #0x28]
100001644: d61f0200    	br	x16
100001648: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000164c: f9401a10    	ldr	x16, [x16, #0x30]
100001650: d61f0200    	br	x16
100001654: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001658: f9401e10    	ldr	x16, [x16, #0x38]
10000165c: d61f0200    	br	x16
100001660: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001664: f9402210    	ldr	x16, [x16, #0x40]
100001668: d61f0200    	br	x16
10000166c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001670: f9402e10    	ldr	x16, [x16, #0x58]
100001674: d61f0200    	br	x16
100001678: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000167c: f9403210    	ldr	x16, [x16, #0x60]
100001680: d61f0200    	br	x16
100001684: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001688: f9403610    	ldr	x16, [x16, #0x68]
10000168c: d61f0200    	br	x16
100001690: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001694: f9403a10    	ldr	x16, [x16, #0x70]
100001698: d61f0200    	br	x16
10000169c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016a0: f9403e10    	ldr	x16, [x16, #0x78]
1000016a4: d61f0200    	br	x16
1000016a8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016ac: f9404610    	ldr	x16, [x16, #0x88]
1000016b0: d61f0200    	br	x16
1000016b4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016b8: f9404a10    	ldr	x16, [x16, #0x90]
1000016bc: d61f0200    	br	x16
1000016c0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016c4: f9405210    	ldr	x16, [x16, #0xa0]
1000016c8: d61f0200    	br	x16
1000016cc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016d0: f9405610    	ldr	x16, [x16, #0xa8]
1000016d4: d61f0200    	br	x16
1000016d8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016dc: f9405a10    	ldr	x16, [x16, #0xb0]
1000016e0: d61f0200    	br	x16
1000016e4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016e8: f9405e10    	ldr	x16, [x16, #0xb8]
1000016ec: d61f0200    	br	x16
1000016f0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000016f4: f9406210    	ldr	x16, [x16, #0xc0]
1000016f8: d61f0200    	br	x16
