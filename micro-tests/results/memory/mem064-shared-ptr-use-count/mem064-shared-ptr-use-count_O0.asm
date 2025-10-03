
/Users/jim/work/cppfort/micro-tests/results/memory/mem064-shared-ptr-use-count/mem064-shared-ptr-use-count_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z25test_shared_ptr_use_countv>:
100000538: d10143ff    	sub	sp, sp, #0x50
10000053c: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000540: 910103fd    	add	x29, sp, #0x40
100000544: d10053a0    	sub	x0, x29, #0x14
100000548: 52800548    	mov	w8, #0x2a               ; =42
10000054c: b81ec3a8    	stur	w8, [x29, #-0x14]
100000550: d10043a8    	sub	x8, x29, #0x10
100000554: f90007e8    	str	x8, [sp, #0x8]
100000558: 94000011    	bl	0x10000059c <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>
10000055c: f94007e1    	ldr	x1, [sp, #0x8]
100000560: 910063e0    	add	x0, sp, #0x18
100000564: f90003e0    	str	x0, [sp]
100000568: 9400001d    	bl	0x1000005dc <__ZNSt3__110shared_ptrIiEC1B8ne200100ERKS1_>
10000056c: f94007e0    	ldr	x0, [sp, #0x8]
100000570: 94000028    	bl	0x100000610 <__ZNKSt3__110shared_ptrIiE9use_countB8ne200100Ev>
100000574: aa0003e8    	mov	x8, x0
100000578: f94003e0    	ldr	x0, [sp]
10000057c: b90017e8    	str	w8, [sp, #0x14]
100000580: 94000039    	bl	0x100000664 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
100000584: f94007e0    	ldr	x0, [sp, #0x8]
100000588: 94000037    	bl	0x100000664 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
10000058c: b94017e0    	ldr	w0, [sp, #0x14]
100000590: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000594: 910143ff    	add	sp, sp, #0x50
100000598: d65f03c0    	ret

000000010000059c <__ZNSt3__111make_sharedB8ne200100IiJiELi0EEENS_10shared_ptrIT_EEDpOT0_>:
10000059c: d10103ff    	sub	sp, sp, #0x40
1000005a0: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000005a4: 9100c3fd    	add	x29, sp, #0x30
1000005a8: f9000be8    	str	x8, [sp, #0x10]
1000005ac: f81f83a8    	stur	x8, [x29, #-0x8]
1000005b0: f81f03a0    	stur	x0, [x29, #-0x10]
1000005b4: d10047a0    	sub	x0, x29, #0x11
1000005b8: f90007e0    	str	x0, [sp, #0x8]
1000005bc: 94000074    	bl	0x10000078c <__ZNSt3__19allocatorIiEC1B8ne200100Ev>
1000005c0: f94007e0    	ldr	x0, [sp, #0x8]
1000005c4: f9400be8    	ldr	x8, [sp, #0x10]
1000005c8: f85f03a1    	ldur	x1, [x29, #-0x10]
1000005cc: 94000039    	bl	0x1000006b0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>
1000005d0: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000005d4: 910103ff    	add	sp, sp, #0x40
1000005d8: d65f03c0    	ret

00000001000005dc <__ZNSt3__110shared_ptrIiEC1B8ne200100ERKS1_>:
1000005dc: d100c3ff    	sub	sp, sp, #0x30
1000005e0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005e4: 910083fd    	add	x29, sp, #0x20
1000005e8: f81f83a0    	stur	x0, [x29, #-0x8]
1000005ec: f9000be1    	str	x1, [sp, #0x10]
1000005f0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000005f4: f90007e0    	str	x0, [sp, #0x8]
1000005f8: f9400be1    	ldr	x1, [sp, #0x10]
1000005fc: 94000372    	bl	0x1000013c4 <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_>
100000600: f94007e0    	ldr	x0, [sp, #0x8]
100000604: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000608: 9100c3ff    	add	sp, sp, #0x30
10000060c: d65f03c0    	ret

0000000100000610 <__ZNKSt3__110shared_ptrIiE9use_countB8ne200100Ev>:
100000610: d100c3ff    	sub	sp, sp, #0x30
100000614: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000618: 910083fd    	add	x29, sp, #0x20
10000061c: f81f83a0    	stur	x0, [x29, #-0x8]
100000620: f85f83a8    	ldur	x8, [x29, #-0x8]
100000624: f9000be8    	str	x8, [sp, #0x10]
100000628: f9400508    	ldr	x8, [x8, #0x8]
10000062c: b40000e8    	cbz	x8, 0x100000648 <__ZNKSt3__110shared_ptrIiE9use_countB8ne200100Ev+0x38>
100000630: 14000001    	b	0x100000634 <__ZNKSt3__110shared_ptrIiE9use_countB8ne200100Ev+0x24>
100000634: f9400be8    	ldr	x8, [sp, #0x10]
100000638: f9400500    	ldr	x0, [x8, #0x8]
10000063c: 9400039b    	bl	0x1000014a8 <__ZNKSt3__119__shared_weak_count9use_countB8ne200100Ev>
100000640: f90007e0    	str	x0, [sp, #0x8]
100000644: 14000004    	b	0x100000654 <__ZNKSt3__110shared_ptrIiE9use_countB8ne200100Ev+0x44>
100000648: d2800008    	mov	x8, #0x0                ; =0
10000064c: f90007e8    	str	x8, [sp, #0x8]
100000650: 14000001    	b	0x100000654 <__ZNKSt3__110shared_ptrIiE9use_countB8ne200100Ev+0x44>
100000654: f94007e0    	ldr	x0, [sp, #0x8]
100000658: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000065c: 9100c3ff    	add	sp, sp, #0x30
100000660: d65f03c0    	ret

0000000100000664 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>:
100000664: d10083ff    	sub	sp, sp, #0x20
100000668: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000066c: 910043fd    	add	x29, sp, #0x10
100000670: f90007e0    	str	x0, [sp, #0x8]
100000674: f94007e0    	ldr	x0, [sp, #0x8]
100000678: f90003e0    	str	x0, [sp]
10000067c: 94000306    	bl	0x100001294 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>
100000680: f94003e0    	ldr	x0, [sp]
100000684: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000688: 910083ff    	add	sp, sp, #0x20
10000068c: d65f03c0    	ret

0000000100000690 <_main>:
100000690: d10083ff    	sub	sp, sp, #0x20
100000694: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000698: 910043fd    	add	x29, sp, #0x10
10000069c: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000006a0: 97ffffa6    	bl	0x100000538 <__Z25test_shared_ptr_use_countv>
1000006a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000006a8: 910083ff    	add	sp, sp, #0x20
1000006ac: d65f03c0    	ret

00000001000006b0 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>:
1000006b0: d10203ff    	sub	sp, sp, #0x80
1000006b4: a9077bfd    	stp	x29, x30, [sp, #0x70]
1000006b8: 9101c3fd    	add	x29, sp, #0x70
1000006bc: f9000be8    	str	x8, [sp, #0x10]
1000006c0: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000006c4: f9400529    	ldr	x9, [x9, #0x8]
1000006c8: f9400129    	ldr	x9, [x9]
1000006cc: f81f83a9    	stur	x9, [x29, #-0x8]
1000006d0: f81d83a8    	stur	x8, [x29, #-0x28]
1000006d4: f81d03a0    	stur	x0, [x29, #-0x30]
1000006d8: f9001fe1    	str	x1, [sp, #0x38]
1000006dc: d10083a0    	sub	x0, x29, #0x20
1000006e0: d2800021    	mov	x1, #0x1                ; =1
1000006e4: 94000035    	bl	0x1000007b8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>
1000006e8: 14000001    	b	0x1000006ec <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x3c>
1000006ec: d10083a0    	sub	x0, x29, #0x20
1000006f0: 9400003f    	bl	0x1000007ec <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>
1000006f4: f9401fe1    	ldr	x1, [sp, #0x38]
1000006f8: 94000043    	bl	0x100000804 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>
1000006fc: 14000001    	b	0x100000700 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x50>
100000700: d10083a0    	sub	x0, x29, #0x20
100000704: f90007e0    	str	x0, [sp, #0x8]
100000708: 9400004c    	bl	0x100000838 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>
10000070c: f9000fe0    	str	x0, [sp, #0x18]
100000710: f9400fe0    	ldr	x0, [sp, #0x18]
100000714: 9400007b    	bl	0x100000900 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000718: f9400be8    	ldr	x8, [sp, #0x10]
10000071c: f9400fe1    	ldr	x1, [sp, #0x18]
100000720: 94000382    	bl	0x100001528 <___stack_chk_guard+0x100001528>
100000724: f94007e0    	ldr	x0, [sp, #0x8]
100000728: 94000080    	bl	0x100000928 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
10000072c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000730: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000734: f9400508    	ldr	x8, [x8, #0x8]
100000738: f9400108    	ldr	x8, [x8]
10000073c: eb090108    	subs	x8, x8, x9
100000740: 54000060    	b.eq	0x10000074c <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x9c>
100000744: 14000001    	b	0x100000748 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x98>
100000748: 9400037b    	bl	0x100001534 <___stack_chk_guard+0x100001534>
10000074c: a9477bfd    	ldp	x29, x30, [sp, #0x70]
100000750: 910203ff    	add	sp, sp, #0x80
100000754: d65f03c0    	ret
100000758: f90017e0    	str	x0, [sp, #0x28]
10000075c: aa0103e8    	mov	x8, x1
100000760: b90027e8    	str	w8, [sp, #0x24]
100000764: d10083a0    	sub	x0, x29, #0x20
100000768: 94000070    	bl	0x100000928 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>
10000076c: 14000001    	b	0x100000770 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xc0>
100000770: f94017e0    	ldr	x0, [sp, #0x28]
100000774: f90003e0    	str	x0, [sp]
100000778: 14000003    	b	0x100000784 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
10000077c: f90003e0    	str	x0, [sp]
100000780: 14000001    	b	0x100000784 <__ZNSt3__115allocate_sharedB8ne200100IiNS_9allocatorIiEEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
100000784: f94003e0    	ldr	x0, [sp]
100000788: 9400036e    	bl	0x100001540 <___stack_chk_guard+0x100001540>

000000010000078c <__ZNSt3__19allocatorIiEC1B8ne200100Ev>:
10000078c: d10083ff    	sub	sp, sp, #0x20
100000790: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000794: 910043fd    	add	x29, sp, #0x10
100000798: f90007e0    	str	x0, [sp, #0x8]
10000079c: f94007e0    	ldr	x0, [sp, #0x8]
1000007a0: f90003e0    	str	x0, [sp]
1000007a4: 940002ac    	bl	0x100001254 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>
1000007a8: f94003e0    	ldr	x0, [sp]
1000007ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000007b0: 910083ff    	add	sp, sp, #0x20
1000007b4: d65f03c0    	ret

00000001000007b8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC1B8ne200100IS3_EET_m>:
1000007b8: d100c3ff    	sub	sp, sp, #0x30
1000007bc: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000007c0: 910083fd    	add	x29, sp, #0x20
1000007c4: f9000be0    	str	x0, [sp, #0x10]
1000007c8: f90007e1    	str	x1, [sp, #0x8]
1000007cc: f9400be0    	ldr	x0, [sp, #0x10]
1000007d0: f90003e0    	str	x0, [sp]
1000007d4: f94007e1    	ldr	x1, [sp, #0x8]
1000007d8: 9400005f    	bl	0x100000954 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>
1000007dc: f94003e0    	ldr	x0, [sp]
1000007e0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000007e4: 9100c3ff    	add	sp, sp, #0x30
1000007e8: d65f03c0    	ret

00000001000007ec <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE5__getB8ne200100Ev>:
1000007ec: d10043ff    	sub	sp, sp, #0x10
1000007f0: f90007e0    	str	x0, [sp, #0x8]
1000007f4: f94007e8    	ldr	x8, [sp, #0x8]
1000007f8: f9400900    	ldr	x0, [x8, #0x10]
1000007fc: 910043ff    	add	sp, sp, #0x10
100000800: d65f03c0    	ret

0000000100000804 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC1B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000804: d100c3ff    	sub	sp, sp, #0x30
100000808: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000080c: 910083fd    	add	x29, sp, #0x20
100000810: f9000be0    	str	x0, [sp, #0x10]
100000814: f90007e1    	str	x1, [sp, #0x8]
100000818: f9400be0    	ldr	x0, [sp, #0x10]
10000081c: f90003e0    	str	x0, [sp]
100000820: f94007e1    	ldr	x1, [sp, #0x8]
100000824: 940000f1    	bl	0x100000be8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>
100000828: f94003e0    	ldr	x0, [sp]
10000082c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000830: 9100c3ff    	add	sp, sp, #0x30
100000834: d65f03c0    	ret

0000000100000838 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE13__release_ptrB8ne200100Ev>:
100000838: d10043ff    	sub	sp, sp, #0x10
10000083c: f90007e0    	str	x0, [sp, #0x8]
100000840: f94007e8    	ldr	x8, [sp, #0x8]
100000844: f9400909    	ldr	x9, [x8, #0x10]
100000848: f90003e9    	str	x9, [sp]
10000084c: f900091f    	str	xzr, [x8, #0x10]
100000850: f94003e0    	ldr	x0, [sp]
100000854: 910043ff    	add	sp, sp, #0x10
100000858: d65f03c0    	ret

000000010000085c <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_>:
10000085c: d10143ff    	sub	sp, sp, #0x50
100000860: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000864: 910103fd    	add	x29, sp, #0x40
100000868: f9000fe8    	str	x8, [sp, #0x18]
10000086c: aa0003e8    	mov	x8, x0
100000870: f9400fe0    	ldr	x0, [sp, #0x18]
100000874: aa0003e9    	mov	x9, x0
100000878: f81f83a9    	stur	x9, [x29, #-0x8]
10000087c: f81f03a8    	stur	x8, [x29, #-0x10]
100000880: f81e83a1    	stur	x1, [x29, #-0x18]
100000884: 52800008    	mov	w8, #0x0                ; =0
100000888: 52800029    	mov	w9, #0x1                ; =1
10000088c: b90023e9    	str	w9, [sp, #0x20]
100000890: 12000108    	and	w8, w8, #0x1
100000894: 12000108    	and	w8, w8, #0x1
100000898: 381e73a8    	sturb	w8, [x29, #-0x19]
10000089c: 94000237    	bl	0x100001178 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>
1000008a0: f9400fe0    	ldr	x0, [sp, #0x18]
1000008a4: f85f03a8    	ldur	x8, [x29, #-0x10]
1000008a8: f9000008    	str	x8, [x0]
1000008ac: f85e83a8    	ldur	x8, [x29, #-0x18]
1000008b0: f9000408    	str	x8, [x0, #0x8]
1000008b4: f940000a    	ldr	x10, [x0]
1000008b8: f9400008    	ldr	x8, [x0]
1000008bc: 910003e9    	mov	x9, sp
1000008c0: f900012a    	str	x10, [x9]
1000008c4: f9000528    	str	x8, [x9, #0x8]
1000008c8: 94000237    	bl	0x1000011a4 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>
1000008cc: b94023e9    	ldr	w9, [sp, #0x20]
1000008d0: 12000128    	and	w8, w9, #0x1
1000008d4: 0a090108    	and	w8, w8, w9
1000008d8: 381e73a8    	sturb	w8, [x29, #-0x19]
1000008dc: 385e73a8    	ldurb	w8, [x29, #-0x19]
1000008e0: 370000a8    	tbnz	w8, #0x0, 0x1000008f4 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
1000008e4: 14000001    	b	0x1000008e8 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x8c>
1000008e8: f9400fe0    	ldr	x0, [sp, #0x18]
1000008ec: 97ffff5e    	bl	0x100000664 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
1000008f0: 14000001    	b	0x1000008f4 <__ZNSt3__110shared_ptrIiE27__create_with_control_blockB8ne200100IiNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEES1_PT_PT0_+0x98>
1000008f4: a9447bfd    	ldp	x29, x30, [sp, #0x40]
1000008f8: 910143ff    	add	sp, sp, #0x50
1000008fc: d65f03c0    	ret

0000000100000900 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>:
100000900: d10083ff    	sub	sp, sp, #0x20
100000904: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000908: 910043fd    	add	x29, sp, #0x10
10000090c: f90007e0    	str	x0, [sp, #0x8]
100000910: f94007e8    	ldr	x8, [sp, #0x8]
100000914: 91006100    	add	x0, x8, #0x18
100000918: 9400022e    	bl	0x1000011d0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>
10000091c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000920: 910083ff    	add	sp, sp, #0x20
100000924: d65f03c0    	ret

0000000100000928 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED1B8ne200100Ev>:
100000928: d10083ff    	sub	sp, sp, #0x20
10000092c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000930: 910043fd    	add	x29, sp, #0x10
100000934: f90007e0    	str	x0, [sp, #0x8]
100000938: f94007e0    	ldr	x0, [sp, #0x8]
10000093c: f90003e0    	str	x0, [sp]
100000940: 94000229    	bl	0x1000011e4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>
100000944: f94003e0    	ldr	x0, [sp]
100000948: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000094c: 910083ff    	add	sp, sp, #0x20
100000950: d65f03c0    	ret

0000000100000954 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100IS3_EET_m>:
100000954: d100c3ff    	sub	sp, sp, #0x30
100000958: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000095c: 910083fd    	add	x29, sp, #0x20
100000960: f9000be0    	str	x0, [sp, #0x10]
100000964: f90007e1    	str	x1, [sp, #0x8]
100000968: f9400be0    	ldr	x0, [sp, #0x10]
10000096c: f90003e0    	str	x0, [sp]
100000970: d10007a1    	sub	x1, x29, #0x1
100000974: 9400000c    	bl	0x1000009a4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000978: f94003e0    	ldr	x0, [sp]
10000097c: f94007e8    	ldr	x8, [sp, #0x8]
100000980: f9000408    	str	x8, [x0, #0x8]
100000984: f9400401    	ldr	x1, [x0, #0x8]
100000988: 94000014    	bl	0x1000009d8 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>
10000098c: aa0003e8    	mov	x8, x0
100000990: f94003e0    	ldr	x0, [sp]
100000994: f9000808    	str	x8, [x0, #0x10]
100000998: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000099c: 9100c3ff    	add	sp, sp, #0x30
1000009a0: d65f03c0    	ret

00000001000009a4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>:
1000009a4: d100c3ff    	sub	sp, sp, #0x30
1000009a8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000009ac: 910083fd    	add	x29, sp, #0x20
1000009b0: f81f83a0    	stur	x0, [x29, #-0x8]
1000009b4: f9000be1    	str	x1, [sp, #0x10]
1000009b8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000009bc: f90007e0    	str	x0, [sp, #0x8]
1000009c0: f9400be1    	ldr	x1, [sp, #0x10]
1000009c4: 94000010    	bl	0x100000a04 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>
1000009c8: f94007e0    	ldr	x0, [sp, #0x8]
1000009cc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000009d0: 9100c3ff    	add	sp, sp, #0x30
1000009d4: d65f03c0    	ret

00000001000009d8 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8allocateB8ne200100ERS5_m>:
1000009d8: d10083ff    	sub	sp, sp, #0x20
1000009dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000009e0: 910043fd    	add	x29, sp, #0x10
1000009e4: f90007e0    	str	x0, [sp, #0x8]
1000009e8: f90003e1    	str	x1, [sp]
1000009ec: f94007e0    	ldr	x0, [sp, #0x8]
1000009f0: f94003e1    	ldr	x1, [sp]
1000009f4: 94000015    	bl	0x100000a48 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>
1000009f8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000009fc: 910083ff    	add	sp, sp, #0x20
100000a00: d65f03c0    	ret

0000000100000a04 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>:
100000a04: d100c3ff    	sub	sp, sp, #0x30
100000a08: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a0c: 910083fd    	add	x29, sp, #0x20
100000a10: f81f83a0    	stur	x0, [x29, #-0x8]
100000a14: f9000be1    	str	x1, [sp, #0x10]
100000a18: f85f83a0    	ldur	x0, [x29, #-0x8]
100000a1c: f90007e0    	str	x0, [sp, #0x8]
100000a20: 94000005    	bl	0x100000a34 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>
100000a24: f94007e0    	ldr	x0, [sp, #0x8]
100000a28: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a2c: 9100c3ff    	add	sp, sp, #0x30
100000a30: d65f03c0    	ret

0000000100000a34 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEEC2B8ne200100Ev>:
100000a34: d10043ff    	sub	sp, sp, #0x10
100000a38: f90007e0    	str	x0, [sp, #0x8]
100000a3c: f94007e0    	ldr	x0, [sp, #0x8]
100000a40: 910043ff    	add	sp, sp, #0x10
100000a44: d65f03c0    	ret

0000000100000a48 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em>:
100000a48: d100c3ff    	sub	sp, sp, #0x30
100000a4c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a50: 910083fd    	add	x29, sp, #0x20
100000a54: f81f83a0    	stur	x0, [x29, #-0x8]
100000a58: f9000be1    	str	x1, [sp, #0x10]
100000a5c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000a60: f9400be8    	ldr	x8, [sp, #0x10]
100000a64: f90007e8    	str	x8, [sp, #0x8]
100000a68: 940002b9    	bl	0x10000154c <___stack_chk_guard+0x10000154c>
100000a6c: f94007e8    	ldr	x8, [sp, #0x8]
100000a70: eb000108    	subs	x8, x8, x0
100000a74: 54000069    	b.ls	0x100000a80 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x38>
100000a78: 14000001    	b	0x100000a7c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE8allocateB8ne200100Em+0x34>
100000a7c: 94000011    	bl	0x100000ac0 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
100000a80: f9400be0    	ldr	x0, [sp, #0x10]
100000a84: d2800101    	mov	x1, #0x8                ; =8
100000a88: 9400001b    	bl	0x100000af4 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>
100000a8c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a90: 9100c3ff    	add	sp, sp, #0x30
100000a94: d65f03c0    	ret

0000000100000a98 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE8max_sizeB8ne200100IS5_vLi0EEEmRKS5_>:
100000a98: d10083ff    	sub	sp, sp, #0x20
100000a9c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000aa0: 910043fd    	add	x29, sp, #0x10
100000aa4: f90007e0    	str	x0, [sp, #0x8]
100000aa8: 9400002e    	bl	0x100000b60 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000aac: d2800408    	mov	x8, #0x20               ; =32
100000ab0: 9ac80800    	udiv	x0, x0, x8
100000ab4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ab8: 910083ff    	add	sp, sp, #0x20
100000abc: d65f03c0    	ret

0000000100000ac0 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000ac0: d10083ff    	sub	sp, sp, #0x20
100000ac4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ac8: 910043fd    	add	x29, sp, #0x10
100000acc: d2800100    	mov	x0, #0x8                ; =8
100000ad0: 940002a2    	bl	0x100001558 <___stack_chk_guard+0x100001558>
100000ad4: f90007e0    	str	x0, [sp, #0x8]
100000ad8: 940002a3    	bl	0x100001564 <___stack_chk_guard+0x100001564>
100000adc: f94007e0    	ldr	x0, [sp, #0x8]
100000ae0: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000ae4: f9402021    	ldr	x1, [x1, #0x40]
100000ae8: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000aec: f9402442    	ldr	x2, [x2, #0x48]
100000af0: 940002a0    	bl	0x100001570 <___stack_chk_guard+0x100001570>

0000000100000af4 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm>:
100000af4: d10103ff    	sub	sp, sp, #0x40
100000af8: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000afc: 9100c3fd    	add	x29, sp, #0x30
100000b00: f81f03a0    	stur	x0, [x29, #-0x10]
100000b04: f9000fe1    	str	x1, [sp, #0x18]
100000b08: f85f03a8    	ldur	x8, [x29, #-0x10]
100000b0c: d37be908    	lsl	x8, x8, #5
100000b10: f9000be8    	str	x8, [sp, #0x10]
100000b14: f9400fe0    	ldr	x0, [sp, #0x18]
100000b18: 94000019    	bl	0x100000b7c <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000b1c: 36000120    	tbz	w0, #0x0, 0x100000b40 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x4c>
100000b20: 14000001    	b	0x100000b24 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x30>
100000b24: f9400fe8    	ldr	x8, [sp, #0x18]
100000b28: f90007e8    	str	x8, [sp, #0x8]
100000b2c: f9400be0    	ldr	x0, [sp, #0x10]
100000b30: f94007e1    	ldr	x1, [sp, #0x8]
100000b34: 94000019    	bl	0x100000b98 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>
100000b38: f81f83a0    	stur	x0, [x29, #-0x8]
100000b3c: 14000005    	b	0x100000b50 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000b40: f9400be0    	ldr	x0, [sp, #0x10]
100000b44: 94000020    	bl	0x100000bc4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>
100000b48: f81f83a0    	stur	x0, [x29, #-0x8]
100000b4c: 14000001    	b	0x100000b50 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPT_NS_15__element_countEm+0x5c>
100000b50: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b54: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000b58: 910103ff    	add	sp, sp, #0x40
100000b5c: d65f03c0    	ret

0000000100000b60 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000b60: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000b64: 910003fd    	mov	x29, sp
100000b68: 94000003    	bl	0x100000b74 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000b6c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000b70: d65f03c0    	ret

0000000100000b74 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000b74: 92800000    	mov	x0, #-0x1               ; =-1
100000b78: d65f03c0    	ret

0000000100000b7c <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000b7c: d10043ff    	sub	sp, sp, #0x10
100000b80: f90007e0    	str	x0, [sp, #0x8]
100000b84: f94007e8    	ldr	x8, [sp, #0x8]
100000b88: f1004108    	subs	x8, x8, #0x10
100000b8c: 1a9f97e0    	cset	w0, hi
100000b90: 910043ff    	add	sp, sp, #0x10
100000b94: d65f03c0    	ret

0000000100000b98 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEJmSt11align_val_tEEEPvDpT0_>:
100000b98: d10083ff    	sub	sp, sp, #0x20
100000b9c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ba0: 910043fd    	add	x29, sp, #0x10
100000ba4: f90007e0    	str	x0, [sp, #0x8]
100000ba8: f90003e1    	str	x1, [sp]
100000bac: f94007e0    	ldr	x0, [sp, #0x8]
100000bb0: f94003e1    	ldr	x1, [sp]
100000bb4: 94000272    	bl	0x10000157c <___stack_chk_guard+0x10000157c>
100000bb8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000bbc: 910083ff    	add	sp, sp, #0x20
100000bc0: d65f03c0    	ret

0000000100000bc4 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEPvm>:
100000bc4: d10083ff    	sub	sp, sp, #0x20
100000bc8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bcc: 910043fd    	add	x29, sp, #0x10
100000bd0: f90007e0    	str	x0, [sp, #0x8]
100000bd4: f94007e0    	ldr	x0, [sp, #0x8]
100000bd8: 9400026c    	bl	0x100001588 <___stack_chk_guard+0x100001588>
100000bdc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000be0: 910083ff    	add	sp, sp, #0x20
100000be4: d65f03c0    	ret

0000000100000be8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_>:
100000be8: d10103ff    	sub	sp, sp, #0x40
100000bec: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000bf0: 9100c3fd    	add	x29, sp, #0x30
100000bf4: f81f03a0    	stur	x0, [x29, #-0x10]
100000bf8: f9000fe1    	str	x1, [sp, #0x18]
100000bfc: f85f03a0    	ldur	x0, [x29, #-0x10]
100000c00: f90003e0    	str	x0, [sp]
100000c04: d2800001    	mov	x1, #0x0                ; =0
100000c08: 94000027    	bl	0x100000ca4 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000c0c: f94003e8    	ldr	x8, [sp]
100000c10: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000c14: 91030129    	add	x9, x9, #0xc0
100000c18: 91004129    	add	x9, x9, #0x10
100000c1c: f9000109    	str	x9, [x8]
100000c20: 91006100    	add	x0, x8, #0x18
100000c24: d10007a1    	sub	x1, x29, #0x1
100000c28: 94000032    	bl	0x100000cf0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>
100000c2c: 14000001    	b	0x100000c30 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x48>
100000c30: f94003e0    	ldr	x0, [sp]
100000c34: 9400003c    	bl	0x100000d24 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000c38: f94003e0    	ldr	x0, [sp]
100000c3c: 97ffff31    	bl	0x100000900 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000c40: aa0003e1    	mov	x1, x0
100000c44: f9400fe2    	ldr	x2, [sp, #0x18]
100000c48: 91002fe0    	add	x0, sp, #0xb
100000c4c: 94000252    	bl	0x100001594 <___stack_chk_guard+0x100001594>
100000c50: 14000001    	b	0x100000c54 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0x6c>
100000c54: f94003e0    	ldr	x0, [sp]
100000c58: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000c5c: 910103ff    	add	sp, sp, #0x40
100000c60: d65f03c0    	ret
100000c64: f9000be0    	str	x0, [sp, #0x10]
100000c68: aa0103e8    	mov	x8, x1
100000c6c: b9000fe8    	str	w8, [sp, #0xc]
100000c70: 14000008    	b	0x100000c90 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000c74: f94003e8    	ldr	x8, [sp]
100000c78: f9000be0    	str	x0, [sp, #0x10]
100000c7c: aa0103e9    	mov	x9, x1
100000c80: b9000fe9    	str	w9, [sp, #0xc]
100000c84: 91006100    	add	x0, x8, #0x18
100000c88: 9400003d    	bl	0x100000d7c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000c8c: 14000001    	b	0x100000c90 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xa8>
100000c90: f94003e0    	ldr	x0, [sp]
100000c94: 94000243    	bl	0x1000015a0 <___stack_chk_guard+0x1000015a0>
100000c98: 14000001    	b	0x100000c9c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEEC2B8ne200100IJiES2_Li0EEES2_DpOT_+0xb4>
100000c9c: f9400be0    	ldr	x0, [sp, #0x10]
100000ca0: 94000228    	bl	0x100001540 <___stack_chk_guard+0x100001540>

0000000100000ca4 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000ca4: d100c3ff    	sub	sp, sp, #0x30
100000ca8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000cac: 910083fd    	add	x29, sp, #0x20
100000cb0: f81f83a0    	stur	x0, [x29, #-0x8]
100000cb4: f9000be1    	str	x1, [sp, #0x10]
100000cb8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000cbc: f90007e0    	str	x0, [sp, #0x8]
100000cc0: f9400be1    	ldr	x1, [sp, #0x10]
100000cc4: 94000070    	bl	0x100000e84 <__ZNSt3__114__shared_countC2B8ne200100El>
100000cc8: f94007e0    	ldr	x0, [sp, #0x8]
100000ccc: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000cd0: f9403d08    	ldr	x8, [x8, #0x78]
100000cd4: 91004108    	add	x8, x8, #0x10
100000cd8: f9000008    	str	x8, [x0]
100000cdc: f9400be8    	ldr	x8, [sp, #0x10]
100000ce0: f9000808    	str	x8, [x0, #0x10]
100000ce4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ce8: 9100c3ff    	add	sp, sp, #0x30
100000cec: d65f03c0    	ret

0000000100000cf0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC1B8ne200100EOS2_>:
100000cf0: d100c3ff    	sub	sp, sp, #0x30
100000cf4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000cf8: 910083fd    	add	x29, sp, #0x20
100000cfc: f81f83a0    	stur	x0, [x29, #-0x8]
100000d00: f9000be1    	str	x1, [sp, #0x10]
100000d04: f85f83a0    	ldur	x0, [x29, #-0x8]
100000d08: f90007e0    	str	x0, [sp, #0x8]
100000d0c: f9400be1    	ldr	x1, [sp, #0x10]
100000d10: 94000069    	bl	0x100000eb4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>
100000d14: f94007e0    	ldr	x0, [sp, #0x8]
100000d18: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000d1c: 9100c3ff    	add	sp, sp, #0x30
100000d20: d65f03c0    	ret

0000000100000d24 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>:
100000d24: d10083ff    	sub	sp, sp, #0x20
100000d28: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d2c: 910043fd    	add	x29, sp, #0x10
100000d30: f90007e0    	str	x0, [sp, #0x8]
100000d34: f94007e8    	ldr	x8, [sp, #0x8]
100000d38: 91006100    	add	x0, x8, #0x18
100000d3c: 9400006a    	bl	0x100000ee4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000d40: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d44: 910083ff    	add	sp, sp, #0x20
100000d48: d65f03c0    	ret

0000000100000d4c <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE9constructB8ne200100IiJiEvLi0EEEvRS2_PT_DpOT0_>:
100000d4c: d100c3ff    	sub	sp, sp, #0x30
100000d50: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000d54: 910083fd    	add	x29, sp, #0x20
100000d58: f81f83a0    	stur	x0, [x29, #-0x8]
100000d5c: f9000be1    	str	x1, [sp, #0x10]
100000d60: f90007e2    	str	x2, [sp, #0x8]
100000d64: f9400be0    	ldr	x0, [sp, #0x10]
100000d68: f94007e1    	ldr	x1, [sp, #0x8]
100000d6c: 94000063    	bl	0x100000ef8 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000d70: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000d74: 9100c3ff    	add	sp, sp, #0x30
100000d78: d65f03c0    	ret

0000000100000d7c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>:
100000d7c: d10083ff    	sub	sp, sp, #0x20
100000d80: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d84: 910043fd    	add	x29, sp, #0x10
100000d88: f90007e0    	str	x0, [sp, #0x8]
100000d8c: f94007e0    	ldr	x0, [sp, #0x8]
100000d90: f90003e0    	str	x0, [sp]
100000d94: 9400006d    	bl	0x100000f48 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>
100000d98: f94003e0    	ldr	x0, [sp]
100000d9c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000da0: 910083ff    	add	sp, sp, #0x20
100000da4: d65f03c0    	ret

0000000100000da8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000da8: d10083ff    	sub	sp, sp, #0x20
100000dac: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000db0: 910043fd    	add	x29, sp, #0x10
100000db4: f90007e0    	str	x0, [sp, #0x8]
100000db8: f94007e0    	ldr	x0, [sp, #0x8]
100000dbc: f90003e0    	str	x0, [sp]
100000dc0: 9400006d    	bl	0x100000f74 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>
100000dc4: f94003e0    	ldr	x0, [sp]
100000dc8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000dcc: 910083ff    	add	sp, sp, #0x20
100000dd0: d65f03c0    	ret

0000000100000dd4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000dd4: d10083ff    	sub	sp, sp, #0x20
100000dd8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ddc: 910043fd    	add	x29, sp, #0x10
100000de0: f90007e0    	str	x0, [sp, #0x8]
100000de4: f94007e0    	ldr	x0, [sp, #0x8]
100000de8: f90003e0    	str	x0, [sp]
100000dec: 97ffffef    	bl	0x100000da8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>
100000df0: f94003e0    	ldr	x0, [sp]
100000df4: 940001ee    	bl	0x1000015ac <___stack_chk_guard+0x1000015ac>
100000df8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000dfc: 910083ff    	add	sp, sp, #0x20
100000e00: d65f03c0    	ret

0000000100000e04 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000e04: d10083ff    	sub	sp, sp, #0x20
100000e08: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e0c: 910043fd    	add	x29, sp, #0x10
100000e10: f90007e0    	str	x0, [sp, #0x8]
100000e14: f94007e0    	ldr	x0, [sp, #0x8]
100000e18: 940001e8    	bl	0x1000015b8 <___stack_chk_guard+0x1000015b8>
100000e1c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e20: 910083ff    	add	sp, sp, #0x20
100000e24: d65f03c0    	ret

0000000100000e28 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000e28: d100c3ff    	sub	sp, sp, #0x30
100000e2c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000e30: 910083fd    	add	x29, sp, #0x20
100000e34: f81f83a0    	stur	x0, [x29, #-0x8]
100000e38: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e3c: f90003e0    	str	x0, [sp]
100000e40: 97ffffb9    	bl	0x100000d24 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000e44: aa0003e1    	mov	x1, x0
100000e48: d10027a0    	sub	x0, x29, #0x9
100000e4c: f90007e0    	str	x0, [sp, #0x8]
100000e50: 97fffed5    	bl	0x1000009a4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000e54: f94003e8    	ldr	x8, [sp]
100000e58: 91006100    	add	x0, x8, #0x18
100000e5c: 97ffffc8    	bl	0x100000d7c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000e60: f94003e0    	ldr	x0, [sp]
100000e64: 94000086    	bl	0x10000107c <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>
100000e68: aa0003e1    	mov	x1, x0
100000e6c: f94007e0    	ldr	x0, [sp, #0x8]
100000e70: d2800022    	mov	x2, #0x1                ; =1
100000e74: 94000075    	bl	0x100001048 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100000e78: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000e7c: 9100c3ff    	add	sp, sp, #0x30
100000e80: d65f03c0    	ret

0000000100000e84 <__ZNSt3__114__shared_countC2B8ne200100El>:
100000e84: d10043ff    	sub	sp, sp, #0x10
100000e88: f90007e0    	str	x0, [sp, #0x8]
100000e8c: f90003e1    	str	x1, [sp]
100000e90: f94007e0    	ldr	x0, [sp, #0x8]
100000e94: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000e98: f9404908    	ldr	x8, [x8, #0x90]
100000e9c: 91004108    	add	x8, x8, #0x10
100000ea0: f9000008    	str	x8, [x0]
100000ea4: f94003e8    	ldr	x8, [sp]
100000ea8: f9000408    	str	x8, [x0, #0x8]
100000eac: 910043ff    	add	sp, sp, #0x10
100000eb0: d65f03c0    	ret

0000000100000eb4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageC2B8ne200100EOS2_>:
100000eb4: d100c3ff    	sub	sp, sp, #0x30
100000eb8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000ebc: 910083fd    	add	x29, sp, #0x20
100000ec0: f81f83a0    	stur	x0, [x29, #-0x8]
100000ec4: f9000be1    	str	x1, [sp, #0x10]
100000ec8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000ecc: f90007e0    	str	x0, [sp, #0x8]
100000ed0: 94000005    	bl	0x100000ee4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000ed4: f94007e0    	ldr	x0, [sp, #0x8]
100000ed8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000edc: 9100c3ff    	add	sp, sp, #0x30
100000ee0: d65f03c0    	ret

0000000100000ee4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>:
100000ee4: d10043ff    	sub	sp, sp, #0x10
100000ee8: f90007e0    	str	x0, [sp, #0x8]
100000eec: f94007e0    	ldr	x0, [sp, #0x8]
100000ef0: 910043ff    	add	sp, sp, #0x10
100000ef4: d65f03c0    	ret

0000000100000ef8 <__ZNSt3__114__construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100000ef8: d10083ff    	sub	sp, sp, #0x20
100000efc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f00: 910043fd    	add	x29, sp, #0x10
100000f04: f90007e0    	str	x0, [sp, #0x8]
100000f08: f90003e1    	str	x1, [sp]
100000f0c: f94007e0    	ldr	x0, [sp, #0x8]
100000f10: f94003e1    	ldr	x1, [sp]
100000f14: 94000004    	bl	0x100000f24 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>
100000f18: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f1c: 910083ff    	add	sp, sp, #0x20
100000f20: d65f03c0    	ret

0000000100000f24 <__ZNSt3__112construct_atB8ne200100IiJiEPiEEPT_S3_DpOT0_>:
100000f24: d10043ff    	sub	sp, sp, #0x10
100000f28: f90007e0    	str	x0, [sp, #0x8]
100000f2c: f90003e1    	str	x1, [sp]
100000f30: f94007e0    	ldr	x0, [sp, #0x8]
100000f34: f94003e8    	ldr	x8, [sp]
100000f38: b9400108    	ldr	w8, [x8]
100000f3c: b9000008    	str	w8, [x0]
100000f40: 910043ff    	add	sp, sp, #0x10
100000f44: d65f03c0    	ret

0000000100000f48 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD2B8ne200100Ev>:
100000f48: d10083ff    	sub	sp, sp, #0x20
100000f4c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f50: 910043fd    	add	x29, sp, #0x10
100000f54: f90007e0    	str	x0, [sp, #0x8]
100000f58: f94007e0    	ldr	x0, [sp, #0x8]
100000f5c: f90003e0    	str	x0, [sp]
100000f60: 97ffffe1    	bl	0x100000ee4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage11__get_allocB8ne200100Ev>
100000f64: f94003e0    	ldr	x0, [sp]
100000f68: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000f6c: 910083ff    	add	sp, sp, #0x20
100000f70: d65f03c0    	ret

0000000100000f74 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED2Ev>:
100000f74: d10083ff    	sub	sp, sp, #0x20
100000f78: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000f7c: 910043fd    	add	x29, sp, #0x10
100000f80: f90007e0    	str	x0, [sp, #0x8]
100000f84: f94007e8    	ldr	x8, [sp, #0x8]
100000f88: f90003e8    	str	x8, [sp]
100000f8c: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000f90: 91030129    	add	x9, x9, #0xc0
100000f94: 91004129    	add	x9, x9, #0x10
100000f98: f9000109    	str	x9, [x8]
100000f9c: 91006100    	add	x0, x8, #0x18
100000fa0: 97ffff77    	bl	0x100000d7c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_StorageD1B8ne200100Ev>
100000fa4: f94003e0    	ldr	x0, [sp]
100000fa8: 9400017e    	bl	0x1000015a0 <___stack_chk_guard+0x1000015a0>
100000fac: f94003e0    	ldr	x0, [sp]
100000fb0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000fb4: 910083ff    	add	sp, sp, #0x20
100000fb8: d65f03c0    	ret

0000000100000fbc <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv>:
100000fbc: d100c3ff    	sub	sp, sp, #0x30
100000fc0: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000fc4: 910083fd    	add	x29, sp, #0x20
100000fc8: f81f83a0    	stur	x0, [x29, #-0x8]
100000fcc: f85f83a0    	ldur	x0, [x29, #-0x8]
100000fd0: f90007e0    	str	x0, [sp, #0x8]
100000fd4: 97ffff54    	bl	0x100000d24 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE11__get_allocB8ne200100Ev>
100000fd8: f94007e0    	ldr	x0, [sp, #0x8]
100000fdc: 97fffe49    	bl	0x100000900 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE10__get_elemB8ne200100Ev>
100000fe0: aa0003e1    	mov	x1, x0
100000fe4: d10027a0    	sub	x0, x29, #0x9
100000fe8: 94000177    	bl	0x1000015c4 <___stack_chk_guard+0x1000015c4>
100000fec: 14000001    	b	0x100000ff0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_implB8ne200100IS2_Li0EEEvv+0x34>
100000ff0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000ff4: 9100c3ff    	add	sp, sp, #0x30
100000ff8: d65f03c0    	ret
100000ffc: 9400000b    	bl	0x100001028 <___clang_call_terminate>

0000000100001000 <__ZNSt3__116allocator_traitsINS_9allocatorIiEEE7destroyB8ne200100IivLi0EEEvRS2_PT_>:
100001000: d10083ff    	sub	sp, sp, #0x20
100001004: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001008: 910043fd    	add	x29, sp, #0x10
10000100c: f90007e0    	str	x0, [sp, #0x8]
100001010: f90003e1    	str	x1, [sp]
100001014: f94003e0    	ldr	x0, [sp]
100001018: 94000008    	bl	0x100001038 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>
10000101c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001020: 910083ff    	add	sp, sp, #0x20
100001024: d65f03c0    	ret

0000000100001028 <___clang_call_terminate>:
100001028: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000102c: 910003fd    	mov	x29, sp
100001030: 94000168    	bl	0x1000015d0 <___stack_chk_guard+0x1000015d0>
100001034: 9400016a    	bl	0x1000015dc <___stack_chk_guard+0x1000015dc>

0000000100001038 <__ZNSt3__112__destroy_atB8ne200100IiLi0EEEvPT_>:
100001038: d10043ff    	sub	sp, sp, #0x10
10000103c: f90007e0    	str	x0, [sp, #0x8]
100001040: 910043ff    	add	sp, sp, #0x10
100001044: d65f03c0    	ret

0000000100001048 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>:
100001048: d100c3ff    	sub	sp, sp, #0x30
10000104c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001050: 910083fd    	add	x29, sp, #0x20
100001054: f81f83a0    	stur	x0, [x29, #-0x8]
100001058: f9000be1    	str	x1, [sp, #0x10]
10000105c: f90007e2    	str	x2, [sp, #0x8]
100001060: f85f83a0    	ldur	x0, [x29, #-0x8]
100001064: f9400be1    	ldr	x1, [sp, #0x10]
100001068: f94007e2    	ldr	x2, [sp, #0x8]
10000106c: 94000009    	bl	0x100001090 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>
100001070: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001074: 9100c3ff    	add	sp, sp, #0x30
100001078: d65f03c0    	ret

000000010000107c <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEE10pointer_toB8ne200100ERS4_>:
10000107c: d10043ff    	sub	sp, sp, #0x10
100001080: f90007e0    	str	x0, [sp, #0x8]
100001084: f94007e0    	ldr	x0, [sp, #0x8]
100001088: 910043ff    	add	sp, sp, #0x10
10000108c: d65f03c0    	ret

0000000100001090 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceIiNS0_IiEEEEE10deallocateB8ne200100EPS3_m>:
100001090: d100c3ff    	sub	sp, sp, #0x30
100001094: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001098: 910083fd    	add	x29, sp, #0x20
10000109c: f81f83a0    	stur	x0, [x29, #-0x8]
1000010a0: f9000be1    	str	x1, [sp, #0x10]
1000010a4: f90007e2    	str	x2, [sp, #0x8]
1000010a8: f9400be0    	ldr	x0, [sp, #0x10]
1000010ac: f94007e1    	ldr	x1, [sp, #0x8]
1000010b0: d2800102    	mov	x2, #0x8                ; =8
1000010b4: 94000004    	bl	0x1000010c4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
1000010b8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000010bc: 9100c3ff    	add	sp, sp, #0x30
1000010c0: d65f03c0    	ret

00000001000010c4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
1000010c4: d10103ff    	sub	sp, sp, #0x40
1000010c8: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000010cc: 9100c3fd    	add	x29, sp, #0x30
1000010d0: f81f83a0    	stur	x0, [x29, #-0x8]
1000010d4: f81f03a1    	stur	x1, [x29, #-0x10]
1000010d8: f9000fe2    	str	x2, [sp, #0x18]
1000010dc: f85f03a8    	ldur	x8, [x29, #-0x10]
1000010e0: d37be908    	lsl	x8, x8, #5
1000010e4: f9000be8    	str	x8, [sp, #0x10]
1000010e8: f9400fe0    	ldr	x0, [sp, #0x18]
1000010ec: 97fffea4    	bl	0x100000b7c <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
1000010f0: 36000100    	tbz	w0, #0x0, 0x100001110 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x4c>
1000010f4: 14000001    	b	0x1000010f8 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x34>
1000010f8: f9400fe8    	ldr	x8, [sp, #0x18]
1000010fc: f90007e8    	str	x8, [sp, #0x8]
100001100: f85f83a0    	ldur	x0, [x29, #-0x8]
100001104: f94007e1    	ldr	x1, [sp, #0x8]
100001108: 94000008    	bl	0x100001128 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>
10000110c: 14000004    	b	0x10000111c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
100001110: f85f83a0    	ldur	x0, [x29, #-0x8]
100001114: 94000010    	bl	0x100001154 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>
100001118: 14000001    	b	0x10000111c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x58>
10000111c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100001120: 910103ff    	add	sp, sp, #0x40
100001124: d65f03c0    	ret

0000000100001128 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEESt11align_val_tEEEvDpT_>:
100001128: d10083ff    	sub	sp, sp, #0x20
10000112c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001130: 910043fd    	add	x29, sp, #0x10
100001134: f90007e0    	str	x0, [sp, #0x8]
100001138: f90003e1    	str	x1, [sp]
10000113c: f94007e0    	ldr	x0, [sp, #0x8]
100001140: f94003e1    	ldr	x1, [sp]
100001144: 94000129    	bl	0x1000015e8 <___stack_chk_guard+0x1000015e8>
100001148: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000114c: 910083ff    	add	sp, sp, #0x20
100001150: d65f03c0    	ret

0000000100001154 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceIiNS_9allocatorIiEEEEEEEvDpT_>:
100001154: d10083ff    	sub	sp, sp, #0x20
100001158: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000115c: 910043fd    	add	x29, sp, #0x10
100001160: f90007e0    	str	x0, [sp, #0x8]
100001164: f94007e0    	ldr	x0, [sp, #0x8]
100001168: 94000111    	bl	0x1000015ac <___stack_chk_guard+0x1000015ac>
10000116c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001170: 910083ff    	add	sp, sp, #0x20
100001174: d65f03c0    	ret

0000000100001178 <__ZNSt3__110shared_ptrIiEC1B8ne200100Ev>:
100001178: d10083ff    	sub	sp, sp, #0x20
10000117c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001180: 910043fd    	add	x29, sp, #0x10
100001184: f90007e0    	str	x0, [sp, #0x8]
100001188: f94007e0    	ldr	x0, [sp, #0x8]
10000118c: f90003e0    	str	x0, [sp]
100001190: 94000009    	bl	0x1000011b4 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>
100001194: f94003e0    	ldr	x0, [sp]
100001198: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000119c: 910083ff    	add	sp, sp, #0x20
1000011a0: d65f03c0    	ret

00000001000011a4 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>:
1000011a4: d10043ff    	sub	sp, sp, #0x10
1000011a8: f90007e0    	str	x0, [sp, #0x8]
1000011ac: 910043ff    	add	sp, sp, #0x10
1000011b0: d65f03c0    	ret

00000001000011b4 <__ZNSt3__110shared_ptrIiEC2B8ne200100Ev>:
1000011b4: d10043ff    	sub	sp, sp, #0x10
1000011b8: f90007e0    	str	x0, [sp, #0x8]
1000011bc: f94007e0    	ldr	x0, [sp, #0x8]
1000011c0: f900001f    	str	xzr, [x0]
1000011c4: f900041f    	str	xzr, [x0, #0x8]
1000011c8: 910043ff    	add	sp, sp, #0x10
1000011cc: d65f03c0    	ret

00000001000011d0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE8_Storage10__get_elemB8ne200100Ev>:
1000011d0: d10043ff    	sub	sp, sp, #0x10
1000011d4: f90007e0    	str	x0, [sp, #0x8]
1000011d8: f94007e0    	ldr	x0, [sp, #0x8]
1000011dc: 910043ff    	add	sp, sp, #0x10
1000011e0: d65f03c0    	ret

00000001000011e4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEED2B8ne200100Ev>:
1000011e4: d10083ff    	sub	sp, sp, #0x20
1000011e8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000011ec: 910043fd    	add	x29, sp, #0x10
1000011f0: f90007e0    	str	x0, [sp, #0x8]
1000011f4: f94007e0    	ldr	x0, [sp, #0x8]
1000011f8: f90003e0    	str	x0, [sp]
1000011fc: 94000005    	bl	0x100001210 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>
100001200: f94003e0    	ldr	x0, [sp]
100001204: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001208: 910083ff    	add	sp, sp, #0x20
10000120c: d65f03c0    	ret

0000000100001210 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev>:
100001210: d10083ff    	sub	sp, sp, #0x20
100001214: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001218: 910043fd    	add	x29, sp, #0x10
10000121c: f90007e0    	str	x0, [sp, #0x8]
100001220: f94007e8    	ldr	x8, [sp, #0x8]
100001224: f90003e8    	str	x8, [sp]
100001228: f9400908    	ldr	x8, [x8, #0x10]
10000122c: b40000e8    	cbz	x8, 0x100001248 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
100001230: 14000001    	b	0x100001234 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x24>
100001234: f94003e0    	ldr	x0, [sp]
100001238: f9400801    	ldr	x1, [x0, #0x10]
10000123c: f9400402    	ldr	x2, [x0, #0x8]
100001240: 97ffff82    	bl	0x100001048 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE10deallocateB8ne200100ERS5_PS4_m>
100001244: 14000001    	b	0x100001248 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceIiNS1_IiEEEEEEE9__destroyB8ne200100Ev+0x38>
100001248: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000124c: 910083ff    	add	sp, sp, #0x20
100001250: d65f03c0    	ret

0000000100001254 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>:
100001254: d10083ff    	sub	sp, sp, #0x20
100001258: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000125c: 910043fd    	add	x29, sp, #0x10
100001260: f90007e0    	str	x0, [sp, #0x8]
100001264: f94007e0    	ldr	x0, [sp, #0x8]
100001268: f90003e0    	str	x0, [sp]
10000126c: 94000005    	bl	0x100001280 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
100001270: f94003e0    	ldr	x0, [sp]
100001274: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001278: 910083ff    	add	sp, sp, #0x20
10000127c: d65f03c0    	ret

0000000100001280 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
100001280: d10043ff    	sub	sp, sp, #0x10
100001284: f90007e0    	str	x0, [sp, #0x8]
100001288: f94007e0    	ldr	x0, [sp, #0x8]
10000128c: 910043ff    	add	sp, sp, #0x10
100001290: d65f03c0    	ret

0000000100001294 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>:
100001294: d100c3ff    	sub	sp, sp, #0x30
100001298: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000129c: 910083fd    	add	x29, sp, #0x20
1000012a0: f9000be0    	str	x0, [sp, #0x10]
1000012a4: f9400be8    	ldr	x8, [sp, #0x10]
1000012a8: f90007e8    	str	x8, [sp, #0x8]
1000012ac: aa0803e9    	mov	x9, x8
1000012b0: f81f83a9    	stur	x9, [x29, #-0x8]
1000012b4: f9400508    	ldr	x8, [x8, #0x8]
1000012b8: b40000c8    	cbz	x8, 0x1000012d0 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
1000012bc: 14000001    	b	0x1000012c0 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x2c>
1000012c0: f94007e8    	ldr	x8, [sp, #0x8]
1000012c4: f9400500    	ldr	x0, [x8, #0x8]
1000012c8: 94000006    	bl	0x1000012e0 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
1000012cc: 14000001    	b	0x1000012d0 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
1000012d0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000012d4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000012d8: 9100c3ff    	add	sp, sp, #0x30
1000012dc: d65f03c0    	ret

00000001000012e0 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
1000012e0: d10083ff    	sub	sp, sp, #0x20
1000012e4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012e8: 910043fd    	add	x29, sp, #0x10
1000012ec: f90007e0    	str	x0, [sp, #0x8]
1000012f0: f94007e0    	ldr	x0, [sp, #0x8]
1000012f4: f90003e0    	str	x0, [sp]
1000012f8: 94000009    	bl	0x10000131c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
1000012fc: 360000a0    	tbz	w0, #0x0, 0x100001310 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
100001300: 14000001    	b	0x100001304 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
100001304: f94003e0    	ldr	x0, [sp]
100001308: 940000bb    	bl	0x1000015f4 <___stack_chk_guard+0x1000015f4>
10000130c: 14000001    	b	0x100001310 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
100001310: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001314: 910083ff    	add	sp, sp, #0x20
100001318: d65f03c0    	ret

000000010000131c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
10000131c: d100c3ff    	sub	sp, sp, #0x30
100001320: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001324: 910083fd    	add	x29, sp, #0x20
100001328: f9000be0    	str	x0, [sp, #0x10]
10000132c: f9400be8    	ldr	x8, [sp, #0x10]
100001330: f90007e8    	str	x8, [sp, #0x8]
100001334: 91002100    	add	x0, x8, #0x8
100001338: 94000017    	bl	0x100001394 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
10000133c: b1000408    	adds	x8, x0, #0x1
100001340: 54000161    	b.ne	0x10000136c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
100001344: 14000001    	b	0x100001348 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
100001348: f94007e0    	ldr	x0, [sp, #0x8]
10000134c: f9400008    	ldr	x8, [x0]
100001350: f9400908    	ldr	x8, [x8, #0x10]
100001354: d63f0100    	blr	x8
100001358: 52800028    	mov	w8, #0x1                ; =1
10000135c: 12000108    	and	w8, w8, #0x1
100001360: 12000108    	and	w8, w8, #0x1
100001364: 381ff3a8    	sturb	w8, [x29, #-0x1]
100001368: 14000006    	b	0x100001380 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
10000136c: 52800008    	mov	w8, #0x0                ; =0
100001370: 12000108    	and	w8, w8, #0x1
100001374: 12000108    	and	w8, w8, #0x1
100001378: 381ff3a8    	sturb	w8, [x29, #-0x1]
10000137c: 14000001    	b	0x100001380 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100001380: 385ff3a8    	ldurb	w8, [x29, #-0x1]
100001384: 12000100    	and	w0, w8, #0x1
100001388: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000138c: 9100c3ff    	add	sp, sp, #0x30
100001390: d65f03c0    	ret

0000000100001394 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
100001394: d10083ff    	sub	sp, sp, #0x20
100001398: f9000fe0    	str	x0, [sp, #0x18]
10000139c: f9400fe8    	ldr	x8, [sp, #0x18]
1000013a0: 92800009    	mov	x9, #-0x1               ; =-1
1000013a4: f9000be9    	str	x9, [sp, #0x10]
1000013a8: f9400be9    	ldr	x9, [sp, #0x10]
1000013ac: f8e90108    	ldaddal	x9, x8, [x8]
1000013b0: 8b090108    	add	x8, x8, x9
1000013b4: f90007e8    	str	x8, [sp, #0x8]
1000013b8: f94007e0    	ldr	x0, [sp, #0x8]
1000013bc: 910083ff    	add	sp, sp, #0x20
1000013c0: d65f03c0    	ret

00000001000013c4 <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_>:
1000013c4: d100c3ff    	sub	sp, sp, #0x30
1000013c8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000013cc: 910083fd    	add	x29, sp, #0x20
1000013d0: f9000be0    	str	x0, [sp, #0x10]
1000013d4: f90007e1    	str	x1, [sp, #0x8]
1000013d8: f9400be8    	ldr	x8, [sp, #0x10]
1000013dc: f90003e8    	str	x8, [sp]
1000013e0: aa0803e9    	mov	x9, x8
1000013e4: f81f83a9    	stur	x9, [x29, #-0x8]
1000013e8: f94007e9    	ldr	x9, [sp, #0x8]
1000013ec: f9400129    	ldr	x9, [x9]
1000013f0: f9000109    	str	x9, [x8]
1000013f4: f94007e9    	ldr	x9, [sp, #0x8]
1000013f8: f9400529    	ldr	x9, [x9, #0x8]
1000013fc: f9000509    	str	x9, [x8, #0x8]
100001400: f9400508    	ldr	x8, [x8, #0x8]
100001404: b40000c8    	cbz	x8, 0x10000141c <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_+0x58>
100001408: 14000001    	b	0x10000140c <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_+0x48>
10000140c: f94003e8    	ldr	x8, [sp]
100001410: f9400500    	ldr	x0, [x8, #0x8]
100001414: 94000006    	bl	0x10000142c <__ZNSt3__119__shared_weak_count12__add_sharedB8ne200100Ev>
100001418: 14000001    	b	0x10000141c <__ZNSt3__110shared_ptrIiEC2B8ne200100ERKS1_+0x58>
10000141c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001420: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001424: 9100c3ff    	add	sp, sp, #0x30
100001428: d65f03c0    	ret

000000010000142c <__ZNSt3__119__shared_weak_count12__add_sharedB8ne200100Ev>:
10000142c: d10083ff    	sub	sp, sp, #0x20
100001430: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001434: 910043fd    	add	x29, sp, #0x10
100001438: f90007e0    	str	x0, [sp, #0x8]
10000143c: f94007e0    	ldr	x0, [sp, #0x8]
100001440: 94000004    	bl	0x100001450 <__ZNSt3__114__shared_count12__add_sharedB8ne200100Ev>
100001444: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001448: 910083ff    	add	sp, sp, #0x20
10000144c: d65f03c0    	ret

0000000100001450 <__ZNSt3__114__shared_count12__add_sharedB8ne200100Ev>:
100001450: d10083ff    	sub	sp, sp, #0x20
100001454: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001458: 910043fd    	add	x29, sp, #0x10
10000145c: f90007e0    	str	x0, [sp, #0x8]
100001460: f94007e8    	ldr	x8, [sp, #0x8]
100001464: 91002100    	add	x0, x8, #0x8
100001468: 94000004    	bl	0x100001478 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>
10000146c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001470: 910083ff    	add	sp, sp, #0x20
100001474: d65f03c0    	ret

0000000100001478 <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>:
100001478: d10083ff    	sub	sp, sp, #0x20
10000147c: f9000fe0    	str	x0, [sp, #0x18]
100001480: f9400fe8    	ldr	x8, [sp, #0x18]
100001484: d2800029    	mov	x9, #0x1                ; =1
100001488: f9000be9    	str	x9, [sp, #0x10]
10000148c: f9400be9    	ldr	x9, [sp, #0x10]
100001490: f8290108    	ldadd	x9, x8, [x8]
100001494: 8b090108    	add	x8, x8, x9
100001498: f90007e8    	str	x8, [sp, #0x8]
10000149c: f94007e0    	ldr	x0, [sp, #0x8]
1000014a0: 910083ff    	add	sp, sp, #0x20
1000014a4: d65f03c0    	ret

00000001000014a8 <__ZNKSt3__119__shared_weak_count9use_countB8ne200100Ev>:
1000014a8: d10083ff    	sub	sp, sp, #0x20
1000014ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000014b0: 910043fd    	add	x29, sp, #0x10
1000014b4: f90007e0    	str	x0, [sp, #0x8]
1000014b8: f94007e0    	ldr	x0, [sp, #0x8]
1000014bc: 94000004    	bl	0x1000014cc <__ZNKSt3__114__shared_count9use_countB8ne200100Ev>
1000014c0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000014c4: 910083ff    	add	sp, sp, #0x20
1000014c8: d65f03c0    	ret

00000001000014cc <__ZNKSt3__114__shared_count9use_countB8ne200100Ev>:
1000014cc: d10083ff    	sub	sp, sp, #0x20
1000014d0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000014d4: 910043fd    	add	x29, sp, #0x10
1000014d8: f90007e0    	str	x0, [sp, #0x8]
1000014dc: f94007e8    	ldr	x8, [sp, #0x8]
1000014e0: 91002100    	add	x0, x8, #0x8
1000014e4: 94000009    	bl	0x100001508 <__ZNSt3__121__libcpp_relaxed_loadB8ne200100IlEET_PKS1_>
1000014e8: f90003e0    	str	x0, [sp]
1000014ec: 14000001    	b	0x1000014f0 <__ZNKSt3__114__shared_count9use_countB8ne200100Ev+0x24>
1000014f0: f94003e8    	ldr	x8, [sp]
1000014f4: 91000500    	add	x0, x8, #0x1
1000014f8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000014fc: 910083ff    	add	sp, sp, #0x20
100001500: d65f03c0    	ret
100001504: 97fffec9    	bl	0x100001028 <___clang_call_terminate>

0000000100001508 <__ZNSt3__121__libcpp_relaxed_loadB8ne200100IlEET_PKS1_>:
100001508: d10043ff    	sub	sp, sp, #0x10
10000150c: f90007e0    	str	x0, [sp, #0x8]
100001510: f94007e8    	ldr	x8, [sp, #0x8]
100001514: f9400108    	ldr	x8, [x8]
100001518: f90003e8    	str	x8, [sp]
10000151c: f94003e0    	ldr	x0, [sp]
100001520: 910043ff    	add	sp, sp, #0x10
100001524: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001528 <__stubs>:
100001528: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000152c: f9400a10    	ldr	x16, [x16, #0x10]
100001530: d61f0200    	br	x16
100001534: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001538: f9400e10    	ldr	x16, [x16, #0x18]
10000153c: d61f0200    	br	x16
100001540: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001544: f9401210    	ldr	x16, [x16, #0x20]
100001548: d61f0200    	br	x16
10000154c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001550: f9401610    	ldr	x16, [x16, #0x28]
100001554: d61f0200    	br	x16
100001558: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000155c: f9401a10    	ldr	x16, [x16, #0x30]
100001560: d61f0200    	br	x16
100001564: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001568: f9401e10    	ldr	x16, [x16, #0x38]
10000156c: d61f0200    	br	x16
100001570: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001574: f9402a10    	ldr	x16, [x16, #0x50]
100001578: d61f0200    	br	x16
10000157c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001580: f9402e10    	ldr	x16, [x16, #0x58]
100001584: d61f0200    	br	x16
100001588: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000158c: f9403210    	ldr	x16, [x16, #0x60]
100001590: d61f0200    	br	x16
100001594: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001598: f9403610    	ldr	x16, [x16, #0x68]
10000159c: d61f0200    	br	x16
1000015a0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015a4: f9403a10    	ldr	x16, [x16, #0x70]
1000015a8: d61f0200    	br	x16
1000015ac: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015b0: f9404210    	ldr	x16, [x16, #0x80]
1000015b4: d61f0200    	br	x16
1000015b8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015bc: f9404610    	ldr	x16, [x16, #0x88]
1000015c0: d61f0200    	br	x16
1000015c4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015c8: f9404e10    	ldr	x16, [x16, #0x98]
1000015cc: d61f0200    	br	x16
1000015d0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015d4: f9405210    	ldr	x16, [x16, #0xa0]
1000015d8: d61f0200    	br	x16
1000015dc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015e0: f9405610    	ldr	x16, [x16, #0xa8]
1000015e4: d61f0200    	br	x16
1000015e8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015ec: f9405a10    	ldr	x16, [x16, #0xb0]
1000015f0: d61f0200    	br	x16
1000015f4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000015f8: f9405e10    	ldr	x16, [x16, #0xb8]
1000015fc: d61f0200    	br	x16
