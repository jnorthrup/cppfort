
/Users/jim/work/cppfort/micro-tests/results/memory/mem069-enable-shared-from-this/mem069-enable-shared-from-this_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <_main>:
100000538: d10183ff    	sub	sp, sp, #0x60
10000053c: a9057bfd    	stp	x29, x30, [sp, #0x50]
100000540: 910143fd    	add	x29, sp, #0x50
100000544: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000548: d10073a0    	sub	x0, x29, #0x1c
10000054c: 52800548    	mov	w8, #0x2a               ; =42
100000550: b81e43a8    	stur	w8, [x29, #-0x1c]
100000554: d10063a8    	sub	x8, x29, #0x18
100000558: f90007e8    	str	x8, [sp, #0x8]
10000055c: 9400001c    	bl	0x1000005cc <__ZNSt3__111make_sharedB8ne200100I4NodeJiELi0EEENS_10shared_ptrIT_EEDpOT0_>
100000560: f94007e0    	ldr	x0, [sp, #0x8]
100000564: 9400002a    	bl	0x10000060c <__ZNKSt3__110shared_ptrI4NodeEptB8ne200100Ev>
100000568: 910083e8    	add	x8, sp, #0x20
10000056c: 94000585    	bl	0x100001b80 <___stack_chk_guard+0x100001b80>
100000570: 14000001    	b	0x100000574 <_main+0x3c>
100000574: 910083e0    	add	x0, sp, #0x20
100000578: f90003e0    	str	x0, [sp]
10000057c: 94000024    	bl	0x10000060c <__ZNKSt3__110shared_ptrI4NodeEptB8ne200100Ev>
100000580: aa0003e8    	mov	x8, x0
100000584: f94003e0    	ldr	x0, [sp]
100000588: b9401108    	ldr	w8, [x8, #0x10]
10000058c: b81fc3a8    	stur	w8, [x29, #-0x4]
100000590: 94000030    	bl	0x100000650 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
100000594: d10063a0    	sub	x0, x29, #0x18
100000598: 9400002e    	bl	0x100000650 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
10000059c: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000005a0: a9457bfd    	ldp	x29, x30, [sp, #0x50]
1000005a4: 910183ff    	add	sp, sp, #0x60
1000005a8: d65f03c0    	ret
1000005ac: f9000fe0    	str	x0, [sp, #0x18]
1000005b0: aa0103e8    	mov	x8, x1
1000005b4: b90017e8    	str	w8, [sp, #0x14]
1000005b8: d10063a0    	sub	x0, x29, #0x18
1000005bc: 94000025    	bl	0x100000650 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
1000005c0: 14000001    	b	0x1000005c4 <_main+0x8c>
1000005c4: f9400fe0    	ldr	x0, [sp, #0x18]
1000005c8: 94000571    	bl	0x100001b8c <___stack_chk_guard+0x100001b8c>

00000001000005cc <__ZNSt3__111make_sharedB8ne200100I4NodeJiELi0EEENS_10shared_ptrIT_EEDpOT0_>:
1000005cc: d10103ff    	sub	sp, sp, #0x40
1000005d0: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000005d4: 9100c3fd    	add	x29, sp, #0x30
1000005d8: f9000be8    	str	x8, [sp, #0x10]
1000005dc: f81f83a8    	stur	x8, [x29, #-0x8]
1000005e0: f81f03a0    	stur	x0, [x29, #-0x10]
1000005e4: d10047a0    	sub	x0, x29, #0x11
1000005e8: f90007e0    	str	x0, [sp, #0x8]
1000005ec: 94000117    	bl	0x100000a48 <__ZNSt3__19allocatorI4NodeEC1B8ne200100Ev>
1000005f0: f94007e0    	ldr	x0, [sp, #0x8]
1000005f4: f9400be8    	ldr	x8, [sp, #0x10]
1000005f8: f85f03a1    	ldur	x1, [x29, #-0x10]
1000005fc: 940000dc    	bl	0x10000096c <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>
100000600: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000604: 910103ff    	add	sp, sp, #0x40
100000608: d65f03c0    	ret

000000010000060c <__ZNKSt3__110shared_ptrI4NodeEptB8ne200100Ev>:
10000060c: d10043ff    	sub	sp, sp, #0x10
100000610: f90007e0    	str	x0, [sp, #0x8]
100000614: f94007e8    	ldr	x8, [sp, #0x8]
100000618: f9400100    	ldr	x0, [x8]
10000061c: 910043ff    	add	sp, sp, #0x10
100000620: d65f03c0    	ret

0000000100000624 <__ZN4Node6getPtrEv>:
100000624: d10083ff    	sub	sp, sp, #0x20
100000628: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000062c: 910043fd    	add	x29, sp, #0x10
100000630: aa0803e9    	mov	x9, x8
100000634: f90007e9    	str	x9, [sp, #0x8]
100000638: f90003e0    	str	x0, [sp]
10000063c: f94003e0    	ldr	x0, [sp]
100000640: 9400000f    	bl	0x10000067c <__ZNSt3__123enable_shared_from_thisI4NodeE16shared_from_thisB8ne200100Ev>
100000644: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000648: 910083ff    	add	sp, sp, #0x20
10000064c: d65f03c0    	ret

0000000100000650 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>:
100000650: d10083ff    	sub	sp, sp, #0x20
100000654: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000658: 910043fd    	add	x29, sp, #0x10
10000065c: f90007e0    	str	x0, [sp, #0x8]
100000660: f94007e0    	ldr	x0, [sp, #0x8]
100000664: f90003e0    	str	x0, [sp]
100000668: 94000075    	bl	0x10000083c <__ZNSt3__110shared_ptrI4NodeED2B8ne200100Ev>
10000066c: f94003e0    	ldr	x0, [sp]
100000670: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000674: 910083ff    	add	sp, sp, #0x20
100000678: d65f03c0    	ret

000000010000067c <__ZNSt3__123enable_shared_from_thisI4NodeE16shared_from_thisB8ne200100Ev>:
10000067c: d100c3ff    	sub	sp, sp, #0x30
100000680: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000684: 910083fd    	add	x29, sp, #0x20
100000688: f90007e8    	str	x8, [sp, #0x8]
10000068c: aa0003e8    	mov	x8, x0
100000690: f94007e0    	ldr	x0, [sp, #0x8]
100000694: aa0003e9    	mov	x9, x0
100000698: f81f83a9    	stur	x9, [x29, #-0x8]
10000069c: f9000be8    	str	x8, [sp, #0x10]
1000006a0: f9400be1    	ldr	x1, [sp, #0x10]
1000006a4: 94000004    	bl	0x1000006b4 <__ZNSt3__110shared_ptrI4NodeEC1B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE>
1000006a8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000006ac: 9100c3ff    	add	sp, sp, #0x30
1000006b0: d65f03c0    	ret

00000001000006b4 <__ZNSt3__110shared_ptrI4NodeEC1B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE>:
1000006b4: d100c3ff    	sub	sp, sp, #0x30
1000006b8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000006bc: 910083fd    	add	x29, sp, #0x20
1000006c0: f81f83a0    	stur	x0, [x29, #-0x8]
1000006c4: f9000be1    	str	x1, [sp, #0x10]
1000006c8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000006cc: f90007e0    	str	x0, [sp, #0x8]
1000006d0: f9400be1    	ldr	x1, [sp, #0x10]
1000006d4: 94000005    	bl	0x1000006e8 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE>
1000006d8: f94007e0    	ldr	x0, [sp, #0x8]
1000006dc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000006e0: 9100c3ff    	add	sp, sp, #0x30
1000006e4: d65f03c0    	ret

00000001000006e8 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE>:
1000006e8: d10103ff    	sub	sp, sp, #0x40
1000006ec: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000006f0: 9100c3fd    	add	x29, sp, #0x30
1000006f4: f81f03a0    	stur	x0, [x29, #-0x10]
1000006f8: f9000fe1    	str	x1, [sp, #0x18]
1000006fc: f85f03a9    	ldur	x9, [x29, #-0x10]
100000700: f9000be9    	str	x9, [sp, #0x10]
100000704: aa0903e8    	mov	x8, x9
100000708: f81f83a8    	stur	x8, [x29, #-0x8]
10000070c: f9400fe8    	ldr	x8, [sp, #0x18]
100000710: f9400108    	ldr	x8, [x8]
100000714: f9000128    	str	x8, [x9]
100000718: f9400fe8    	ldr	x8, [sp, #0x18]
10000071c: f9400508    	ldr	x8, [x8, #0x8]
100000720: b40000e8    	cbz	x8, 0x10000073c <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE+0x54>
100000724: 14000001    	b	0x100000728 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE+0x40>
100000728: f9400fe8    	ldr	x8, [sp, #0x18]
10000072c: f9400500    	ldr	x0, [x8, #0x8]
100000730: 9400051a    	bl	0x100001b98 <___stack_chk_guard+0x100001b98>
100000734: f90007e0    	str	x0, [sp, #0x8]
100000738: 14000005    	b	0x10000074c <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE+0x64>
10000073c: f9400fe8    	ldr	x8, [sp, #0x18]
100000740: f9400508    	ldr	x8, [x8, #0x8]
100000744: f90007e8    	str	x8, [sp, #0x8]
100000748: 14000001    	b	0x10000074c <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE+0x64>
10000074c: f9400be8    	ldr	x8, [sp, #0x10]
100000750: f94007e9    	ldr	x9, [sp, #0x8]
100000754: f9000509    	str	x9, [x8, #0x8]
100000758: f9400508    	ldr	x8, [x8, #0x8]
10000075c: b5000068    	cbnz	x8, 0x100000768 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE+0x80>
100000760: 14000001    	b	0x100000764 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_Li0EEERKNS_8weak_ptrIT_EE+0x7c>
100000764: 94000005    	bl	0x100000778 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>
100000768: f85f83a0    	ldur	x0, [x29, #-0x8]
10000076c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000770: 910103ff    	add	sp, sp, #0x40
100000774: d65f03c0    	ret

0000000100000778 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>:
100000778: d10083ff    	sub	sp, sp, #0x20
10000077c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000780: 910043fd    	add	x29, sp, #0x10
100000784: d2800100    	mov	x0, #0x8                ; =8
100000788: 94000507    	bl	0x100001ba4 <___stack_chk_guard+0x100001ba4>
10000078c: f90007e0    	str	x0, [sp, #0x8]
100000790: f900001f    	str	xzr, [x0]
100000794: 94000007    	bl	0x1000007b0 <__ZNSt3__112bad_weak_ptrC1B8ne200100Ev>
100000798: f94007e0    	ldr	x0, [sp, #0x8]
10000079c: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
1000007a0: f9401421    	ldr	x1, [x1, #0x28]
1000007a4: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
1000007a8: f9401842    	ldr	x2, [x2, #0x30]
1000007ac: 94000501    	bl	0x100001bb0 <___stack_chk_guard+0x100001bb0>

00000001000007b0 <__ZNSt3__112bad_weak_ptrC1B8ne200100Ev>:
1000007b0: d10083ff    	sub	sp, sp, #0x20
1000007b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000007b8: 910043fd    	add	x29, sp, #0x10
1000007bc: f90007e0    	str	x0, [sp, #0x8]
1000007c0: f94007e0    	ldr	x0, [sp, #0x8]
1000007c4: f90003e0    	str	x0, [sp]
1000007c8: 94000005    	bl	0x1000007dc <__ZNSt3__112bad_weak_ptrC2B8ne200100Ev>
1000007cc: f94003e0    	ldr	x0, [sp]
1000007d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000007d4: 910083ff    	add	sp, sp, #0x20
1000007d8: d65f03c0    	ret

00000001000007dc <__ZNSt3__112bad_weak_ptrC2B8ne200100Ev>:
1000007dc: d10083ff    	sub	sp, sp, #0x20
1000007e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000007e4: 910043fd    	add	x29, sp, #0x10
1000007e8: f90007e0    	str	x0, [sp, #0x8]
1000007ec: f94007e0    	ldr	x0, [sp, #0x8]
1000007f0: f90003e0    	str	x0, [sp]
1000007f4: 94000009    	bl	0x100000818 <__ZNSt9exceptionC2B8ne200100Ev>
1000007f8: f94003e0    	ldr	x0, [sp]
1000007fc: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000800: f9402108    	ldr	x8, [x8, #0x40]
100000804: 91004108    	add	x8, x8, #0x10
100000808: f9000008    	str	x8, [x0]
10000080c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000810: 910083ff    	add	sp, sp, #0x20
100000814: d65f03c0    	ret

0000000100000818 <__ZNSt9exceptionC2B8ne200100Ev>:
100000818: d10043ff    	sub	sp, sp, #0x10
10000081c: f90007e0    	str	x0, [sp, #0x8]
100000820: f94007e0    	ldr	x0, [sp, #0x8]
100000824: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000828: f9402508    	ldr	x8, [x8, #0x48]
10000082c: 91004108    	add	x8, x8, #0x10
100000830: f9000008    	str	x8, [x0]
100000834: 910043ff    	add	sp, sp, #0x10
100000838: d65f03c0    	ret

000000010000083c <__ZNSt3__110shared_ptrI4NodeED2B8ne200100Ev>:
10000083c: d100c3ff    	sub	sp, sp, #0x30
100000840: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000844: 910083fd    	add	x29, sp, #0x20
100000848: f9000be0    	str	x0, [sp, #0x10]
10000084c: f9400be8    	ldr	x8, [sp, #0x10]
100000850: f90007e8    	str	x8, [sp, #0x8]
100000854: aa0803e9    	mov	x9, x8
100000858: f81f83a9    	stur	x9, [x29, #-0x8]
10000085c: f9400508    	ldr	x8, [x8, #0x8]
100000860: b40000c8    	cbz	x8, 0x100000878 <__ZNSt3__110shared_ptrI4NodeED2B8ne200100Ev+0x3c>
100000864: 14000001    	b	0x100000868 <__ZNSt3__110shared_ptrI4NodeED2B8ne200100Ev+0x2c>
100000868: f94007e8    	ldr	x8, [sp, #0x8]
10000086c: f9400500    	ldr	x0, [x8, #0x8]
100000870: 94000006    	bl	0x100000888 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
100000874: 14000001    	b	0x100000878 <__ZNSt3__110shared_ptrI4NodeED2B8ne200100Ev+0x3c>
100000878: f85f83a0    	ldur	x0, [x29, #-0x8]
10000087c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000880: 9100c3ff    	add	sp, sp, #0x30
100000884: d65f03c0    	ret

0000000100000888 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
100000888: d10083ff    	sub	sp, sp, #0x20
10000088c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000890: 910043fd    	add	x29, sp, #0x10
100000894: f90007e0    	str	x0, [sp, #0x8]
100000898: f94007e0    	ldr	x0, [sp, #0x8]
10000089c: f90003e0    	str	x0, [sp]
1000008a0: 94000009    	bl	0x1000008c4 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
1000008a4: 360000a0    	tbz	w0, #0x0, 0x1000008b8 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
1000008a8: 14000001    	b	0x1000008ac <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
1000008ac: f94003e0    	ldr	x0, [sp]
1000008b0: 940004c3    	bl	0x100001bbc <___stack_chk_guard+0x100001bbc>
1000008b4: 14000001    	b	0x1000008b8 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
1000008b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000008bc: 910083ff    	add	sp, sp, #0x20
1000008c0: d65f03c0    	ret

00000001000008c4 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
1000008c4: d100c3ff    	sub	sp, sp, #0x30
1000008c8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000008cc: 910083fd    	add	x29, sp, #0x20
1000008d0: f9000be0    	str	x0, [sp, #0x10]
1000008d4: f9400be8    	ldr	x8, [sp, #0x10]
1000008d8: f90007e8    	str	x8, [sp, #0x8]
1000008dc: 91002100    	add	x0, x8, #0x8
1000008e0: 94000017    	bl	0x10000093c <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
1000008e4: b1000408    	adds	x8, x0, #0x1
1000008e8: 54000161    	b.ne	0x100000914 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
1000008ec: 14000001    	b	0x1000008f0 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
1000008f0: f94007e0    	ldr	x0, [sp, #0x8]
1000008f4: f9400008    	ldr	x8, [x0]
1000008f8: f9400908    	ldr	x8, [x8, #0x10]
1000008fc: d63f0100    	blr	x8
100000900: 52800028    	mov	w8, #0x1                ; =1
100000904: 12000108    	and	w8, w8, #0x1
100000908: 12000108    	and	w8, w8, #0x1
10000090c: 381ff3a8    	sturb	w8, [x29, #-0x1]
100000910: 14000006    	b	0x100000928 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100000914: 52800008    	mov	w8, #0x0                ; =0
100000918: 12000108    	and	w8, w8, #0x1
10000091c: 12000108    	and	w8, w8, #0x1
100000920: 381ff3a8    	sturb	w8, [x29, #-0x1]
100000924: 14000001    	b	0x100000928 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100000928: 385ff3a8    	ldurb	w8, [x29, #-0x1]
10000092c: 12000100    	and	w0, w8, #0x1
100000930: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000934: 9100c3ff    	add	sp, sp, #0x30
100000938: d65f03c0    	ret

000000010000093c <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
10000093c: d10083ff    	sub	sp, sp, #0x20
100000940: f9000fe0    	str	x0, [sp, #0x18]
100000944: f9400fe8    	ldr	x8, [sp, #0x18]
100000948: 92800009    	mov	x9, #-0x1               ; =-1
10000094c: f9000be9    	str	x9, [sp, #0x10]
100000950: f9400be9    	ldr	x9, [sp, #0x10]
100000954: f8e90108    	ldaddal	x9, x8, [x8]
100000958: 8b090108    	add	x8, x8, x9
10000095c: f90007e8    	str	x8, [sp, #0x8]
100000960: f94007e0    	ldr	x0, [sp, #0x8]
100000964: 910083ff    	add	sp, sp, #0x20
100000968: d65f03c0    	ret

000000010000096c <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_>:
10000096c: d10203ff    	sub	sp, sp, #0x80
100000970: a9077bfd    	stp	x29, x30, [sp, #0x70]
100000974: 9101c3fd    	add	x29, sp, #0x70
100000978: f9000be8    	str	x8, [sp, #0x10]
10000097c: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000980: f9402d29    	ldr	x9, [x9, #0x58]
100000984: f9400129    	ldr	x9, [x9]
100000988: f81f83a9    	stur	x9, [x29, #-0x8]
10000098c: f81d83a8    	stur	x8, [x29, #-0x28]
100000990: f81d03a0    	stur	x0, [x29, #-0x30]
100000994: f9001fe1    	str	x1, [sp, #0x38]
100000998: d10083a0    	sub	x0, x29, #0x20
10000099c: d2800021    	mov	x1, #0x1                ; =1
1000009a0: 94000035    	bl	0x100000a74 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEEC1B8ne200100IS4_EET_m>
1000009a4: 14000001    	b	0x1000009a8 <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x3c>
1000009a8: d10083a0    	sub	x0, x29, #0x20
1000009ac: 9400003f    	bl	0x100000aa8 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE5__getB8ne200100Ev>
1000009b0: f9401fe1    	ldr	x1, [sp, #0x38]
1000009b4: 94000043    	bl	0x100000ac0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC1B8ne200100IJiES3_Li0EEES3_DpOT_>
1000009b8: 14000001    	b	0x1000009bc <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x50>
1000009bc: d10083a0    	sub	x0, x29, #0x20
1000009c0: f90007e0    	str	x0, [sp, #0x8]
1000009c4: 9400004c    	bl	0x100000af4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE13__release_ptrB8ne200100Ev>
1000009c8: f9000fe0    	str	x0, [sp, #0x18]
1000009cc: f9400fe0    	ldr	x0, [sp, #0x18]
1000009d0: 94000078    	bl	0x100000bb0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE10__get_elemB8ne200100Ev>
1000009d4: f9400be8    	ldr	x8, [sp, #0x10]
1000009d8: f9400fe1    	ldr	x1, [sp, #0x18]
1000009dc: 9400047b    	bl	0x100001bc8 <___stack_chk_guard+0x100001bc8>
1000009e0: f94007e0    	ldr	x0, [sp, #0x8]
1000009e4: 9400007d    	bl	0x100000bd8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEED1B8ne200100Ev>
1000009e8: f85f83a9    	ldur	x9, [x29, #-0x8]
1000009ec: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000009f0: f9402d08    	ldr	x8, [x8, #0x58]
1000009f4: f9400108    	ldr	x8, [x8]
1000009f8: eb090108    	subs	x8, x8, x9
1000009fc: 54000060    	b.eq	0x100000a08 <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x9c>
100000a00: 14000001    	b	0x100000a04 <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0x98>
100000a04: 94000474    	bl	0x100001bd4 <___stack_chk_guard+0x100001bd4>
100000a08: a9477bfd    	ldp	x29, x30, [sp, #0x70]
100000a0c: 910203ff    	add	sp, sp, #0x80
100000a10: d65f03c0    	ret
100000a14: f90017e0    	str	x0, [sp, #0x28]
100000a18: aa0103e8    	mov	x8, x1
100000a1c: b90027e8    	str	w8, [sp, #0x24]
100000a20: d10083a0    	sub	x0, x29, #0x20
100000a24: 9400006d    	bl	0x100000bd8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEED1B8ne200100Ev>
100000a28: 14000001    	b	0x100000a2c <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xc0>
100000a2c: f94017e0    	ldr	x0, [sp, #0x28]
100000a30: f90003e0    	str	x0, [sp]
100000a34: 14000003    	b	0x100000a40 <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
100000a38: f90003e0    	str	x0, [sp]
100000a3c: 14000001    	b	0x100000a40 <__ZNSt3__115allocate_sharedB8ne200100I4NodeNS_9allocatorIS1_EEJiELi0EEENS_10shared_ptrIT_EERKT0_DpOT1_+0xd4>
100000a40: f94003e0    	ldr	x0, [sp]
100000a44: 94000452    	bl	0x100001b8c <___stack_chk_guard+0x100001b8c>

0000000100000a48 <__ZNSt3__19allocatorI4NodeEC1B8ne200100Ev>:
100000a48: d10083ff    	sub	sp, sp, #0x20
100000a4c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a50: 910043fd    	add	x29, sp, #0x10
100000a54: f90007e0    	str	x0, [sp, #0x8]
100000a58: f94007e0    	ldr	x0, [sp, #0x8]
100000a5c: f90003e0    	str	x0, [sp]
100000a60: 94000438    	bl	0x100001b40 <__ZNSt3__19allocatorI4NodeEC2B8ne200100Ev>
100000a64: f94003e0    	ldr	x0, [sp]
100000a68: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a6c: 910083ff    	add	sp, sp, #0x20
100000a70: d65f03c0    	ret

0000000100000a74 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEEC1B8ne200100IS4_EET_m>:
100000a74: d100c3ff    	sub	sp, sp, #0x30
100000a78: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000a7c: 910083fd    	add	x29, sp, #0x20
100000a80: f9000be0    	str	x0, [sp, #0x10]
100000a84: f90007e1    	str	x1, [sp, #0x8]
100000a88: f9400be0    	ldr	x0, [sp, #0x10]
100000a8c: f90003e0    	str	x0, [sp]
100000a90: f94007e1    	ldr	x1, [sp, #0x8]
100000a94: 9400005c    	bl	0x100000c04 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEEC2B8ne200100IS4_EET_m>
100000a98: f94003e0    	ldr	x0, [sp]
100000a9c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000aa0: 9100c3ff    	add	sp, sp, #0x30
100000aa4: d65f03c0    	ret

0000000100000aa8 <__ZNKSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE5__getB8ne200100Ev>:
100000aa8: d10043ff    	sub	sp, sp, #0x10
100000aac: f90007e0    	str	x0, [sp, #0x8]
100000ab0: f94007e8    	ldr	x8, [sp, #0x8]
100000ab4: f9400900    	ldr	x0, [x8, #0x10]
100000ab8: 910043ff    	add	sp, sp, #0x10
100000abc: d65f03c0    	ret

0000000100000ac0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC1B8ne200100IJiES3_Li0EEES3_DpOT_>:
100000ac0: d100c3ff    	sub	sp, sp, #0x30
100000ac4: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000ac8: 910083fd    	add	x29, sp, #0x20
100000acc: f9000be0    	str	x0, [sp, #0x10]
100000ad0: f90007e1    	str	x1, [sp, #0x8]
100000ad4: f9400be0    	ldr	x0, [sp, #0x10]
100000ad8: f90003e0    	str	x0, [sp]
100000adc: f94007e1    	ldr	x1, [sp, #0x8]
100000ae0: 940000ef    	bl	0x100000e9c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_>
100000ae4: f94003e0    	ldr	x0, [sp]
100000ae8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000aec: 9100c3ff    	add	sp, sp, #0x30
100000af0: d65f03c0    	ret

0000000100000af4 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE13__release_ptrB8ne200100Ev>:
100000af4: d10043ff    	sub	sp, sp, #0x10
100000af8: f90007e0    	str	x0, [sp, #0x8]
100000afc: f94007e8    	ldr	x8, [sp, #0x8]
100000b00: f9400909    	ldr	x9, [x8, #0x10]
100000b04: f90003e9    	str	x9, [sp]
100000b08: f900091f    	str	xzr, [x8, #0x10]
100000b0c: f94003e0    	ldr	x0, [sp]
100000b10: 910043ff    	add	sp, sp, #0x10
100000b14: d65f03c0    	ret

0000000100000b18 <__ZNSt3__110shared_ptrI4NodeE27__create_with_control_blockB8ne200100IS1_NS_20__shared_ptr_emplaceIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_>:
100000b18: d10103ff    	sub	sp, sp, #0x40
100000b1c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000b20: 9100c3fd    	add	x29, sp, #0x30
100000b24: f90007e8    	str	x8, [sp, #0x8]
100000b28: aa0003e8    	mov	x8, x0
100000b2c: f94007e0    	ldr	x0, [sp, #0x8]
100000b30: aa0003e9    	mov	x9, x0
100000b34: f81f83a9    	stur	x9, [x29, #-0x8]
100000b38: f81f03a8    	stur	x8, [x29, #-0x10]
100000b3c: f9000fe1    	str	x1, [sp, #0x18]
100000b40: 52800008    	mov	w8, #0x0                ; =0
100000b44: 52800029    	mov	w9, #0x1                ; =1
100000b48: b90013e9    	str	w9, [sp, #0x10]
100000b4c: 12000108    	and	w8, w8, #0x1
100000b50: 12000108    	and	w8, w8, #0x1
100000b54: 39005fe8    	strb	w8, [sp, #0x17]
100000b58: 940002b7    	bl	0x100001634 <__ZNSt3__110shared_ptrI4NodeEC1B8ne200100Ev>
100000b5c: f94007e0    	ldr	x0, [sp, #0x8]
100000b60: f85f03a8    	ldur	x8, [x29, #-0x10]
100000b64: f9000008    	str	x8, [x0]
100000b68: f9400fe8    	ldr	x8, [sp, #0x18]
100000b6c: f9000408    	str	x8, [x0, #0x8]
100000b70: f9400001    	ldr	x1, [x0]
100000b74: f9400002    	ldr	x2, [x0]
100000b78: 9400041a    	bl	0x100001be0 <___stack_chk_guard+0x100001be0>
100000b7c: b94013e9    	ldr	w9, [sp, #0x10]
100000b80: 12000128    	and	w8, w9, #0x1
100000b84: 0a090108    	and	w8, w8, w9
100000b88: 39005fe8    	strb	w8, [sp, #0x17]
100000b8c: 39405fe8    	ldrb	w8, [sp, #0x17]
100000b90: 370000a8    	tbnz	w8, #0x0, 0x100000ba4 <__ZNSt3__110shared_ptrI4NodeE27__create_with_control_blockB8ne200100IS1_NS_20__shared_ptr_emplaceIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_+0x8c>
100000b94: 14000001    	b	0x100000b98 <__ZNSt3__110shared_ptrI4NodeE27__create_with_control_blockB8ne200100IS1_NS_20__shared_ptr_emplaceIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_+0x80>
100000b98: f94007e0    	ldr	x0, [sp, #0x8]
100000b9c: 97fffead    	bl	0x100000650 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
100000ba0: 14000001    	b	0x100000ba4 <__ZNSt3__110shared_ptrI4NodeE27__create_with_control_blockB8ne200100IS1_NS_20__shared_ptr_emplaceIS1_NS_9allocatorIS1_EEEEEES2_PT_PT0_+0x8c>
100000ba4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000ba8: 910103ff    	add	sp, sp, #0x40
100000bac: d65f03c0    	ret

0000000100000bb0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE10__get_elemB8ne200100Ev>:
100000bb0: d10083ff    	sub	sp, sp, #0x20
100000bb4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000bb8: 910043fd    	add	x29, sp, #0x10
100000bbc: f90007e0    	str	x0, [sp, #0x8]
100000bc0: f94007e8    	ldr	x8, [sp, #0x8]
100000bc4: 91006100    	add	x0, x8, #0x18
100000bc8: 940003bd    	bl	0x100001abc <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_Storage10__get_elemB8ne200100Ev>
100000bcc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000bd0: 910083ff    	add	sp, sp, #0x20
100000bd4: d65f03c0    	ret

0000000100000bd8 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEED1B8ne200100Ev>:
100000bd8: d10083ff    	sub	sp, sp, #0x20
100000bdc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000be0: 910043fd    	add	x29, sp, #0x10
100000be4: f90007e0    	str	x0, [sp, #0x8]
100000be8: f94007e0    	ldr	x0, [sp, #0x8]
100000bec: f90003e0    	str	x0, [sp]
100000bf0: 940003b8    	bl	0x100001ad0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEED2B8ne200100Ev>
100000bf4: f94003e0    	ldr	x0, [sp]
100000bf8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000bfc: 910083ff    	add	sp, sp, #0x20
100000c00: d65f03c0    	ret

0000000100000c04 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEEC2B8ne200100IS4_EET_m>:
100000c04: d100c3ff    	sub	sp, sp, #0x30
100000c08: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c0c: 910083fd    	add	x29, sp, #0x20
100000c10: f9000be0    	str	x0, [sp, #0x10]
100000c14: f90007e1    	str	x1, [sp, #0x8]
100000c18: f9400be0    	ldr	x0, [sp, #0x10]
100000c1c: f90003e0    	str	x0, [sp]
100000c20: d10007a1    	sub	x1, x29, #0x1
100000c24: 9400000c    	bl	0x100000c54 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEEC1B8ne200100IS2_EERKNS0_IT_EE>
100000c28: f94003e0    	ldr	x0, [sp]
100000c2c: f94007e8    	ldr	x8, [sp, #0x8]
100000c30: f9000408    	str	x8, [x0, #0x8]
100000c34: f9400401    	ldr	x1, [x0, #0x8]
100000c38: 94000014    	bl	0x100000c88 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE8allocateB8ne200100ERS6_m>
100000c3c: aa0003e8    	mov	x8, x0
100000c40: f94003e0    	ldr	x0, [sp]
100000c44: f9000808    	str	x8, [x0, #0x10]
100000c48: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c4c: 9100c3ff    	add	sp, sp, #0x30
100000c50: d65f03c0    	ret

0000000100000c54 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEEC1B8ne200100IS2_EERKNS0_IT_EE>:
100000c54: d100c3ff    	sub	sp, sp, #0x30
100000c58: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000c5c: 910083fd    	add	x29, sp, #0x20
100000c60: f81f83a0    	stur	x0, [x29, #-0x8]
100000c64: f9000be1    	str	x1, [sp, #0x10]
100000c68: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c6c: f90007e0    	str	x0, [sp, #0x8]
100000c70: f9400be1    	ldr	x1, [sp, #0x10]
100000c74: 94000010    	bl	0x100000cb4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEEC2B8ne200100IS2_EERKNS0_IT_EE>
100000c78: f94007e0    	ldr	x0, [sp, #0x8]
100000c7c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c80: 9100c3ff    	add	sp, sp, #0x30
100000c84: d65f03c0    	ret

0000000100000c88 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE8allocateB8ne200100ERS6_m>:
100000c88: d10083ff    	sub	sp, sp, #0x20
100000c8c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000c90: 910043fd    	add	x29, sp, #0x10
100000c94: f90007e0    	str	x0, [sp, #0x8]
100000c98: f90003e1    	str	x1, [sp]
100000c9c: f94007e0    	ldr	x0, [sp, #0x8]
100000ca0: f94003e1    	ldr	x1, [sp]
100000ca4: 94000015    	bl	0x100000cf8 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEE8allocateB8ne200100Em>
100000ca8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cac: 910083ff    	add	sp, sp, #0x20
100000cb0: d65f03c0    	ret

0000000100000cb4 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEEC2B8ne200100IS2_EERKNS0_IT_EE>:
100000cb4: d100c3ff    	sub	sp, sp, #0x30
100000cb8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000cbc: 910083fd    	add	x29, sp, #0x20
100000cc0: f81f83a0    	stur	x0, [x29, #-0x8]
100000cc4: f9000be1    	str	x1, [sp, #0x10]
100000cc8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000ccc: f90007e0    	str	x0, [sp, #0x8]
100000cd0: 94000005    	bl	0x100000ce4 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEEC2B8ne200100Ev>
100000cd4: f94007e0    	ldr	x0, [sp, #0x8]
100000cd8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000cdc: 9100c3ff    	add	sp, sp, #0x30
100000ce0: d65f03c0    	ret

0000000100000ce4 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEEC2B8ne200100Ev>:
100000ce4: d10043ff    	sub	sp, sp, #0x10
100000ce8: f90007e0    	str	x0, [sp, #0x8]
100000cec: f94007e0    	ldr	x0, [sp, #0x8]
100000cf0: 910043ff    	add	sp, sp, #0x10
100000cf4: d65f03c0    	ret

0000000100000cf8 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEE8allocateB8ne200100Em>:
100000cf8: d100c3ff    	sub	sp, sp, #0x30
100000cfc: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000d00: 910083fd    	add	x29, sp, #0x20
100000d04: f81f83a0    	stur	x0, [x29, #-0x8]
100000d08: f9000be1    	str	x1, [sp, #0x10]
100000d0c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000d10: f9400be8    	ldr	x8, [sp, #0x10]
100000d14: f90007e8    	str	x8, [sp, #0x8]
100000d18: 940003b5    	bl	0x100001bec <___stack_chk_guard+0x100001bec>
100000d1c: f94007e8    	ldr	x8, [sp, #0x8]
100000d20: eb000108    	subs	x8, x8, x0
100000d24: 54000069    	b.ls	0x100000d30 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEE8allocateB8ne200100Em+0x38>
100000d28: 14000001    	b	0x100000d2c <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEE8allocateB8ne200100Em+0x34>
100000d2c: 94000011    	bl	0x100000d70 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>
100000d30: f9400be0    	ldr	x0, [sp, #0x10]
100000d34: d2800101    	mov	x1, #0x8                ; =8
100000d38: 9400001b    	bl	0x100000da4 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPT_NS_15__element_countEm>
100000d3c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000d40: 9100c3ff    	add	sp, sp, #0x30
100000d44: d65f03c0    	ret

0000000100000d48 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE8max_sizeB8ne200100IS6_vLi0EEEmRKS6_>:
100000d48: d10083ff    	sub	sp, sp, #0x20
100000d4c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d50: 910043fd    	add	x29, sp, #0x10
100000d54: f90007e0    	str	x0, [sp, #0x8]
100000d58: 9400002f    	bl	0x100000e14 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>
100000d5c: d2800608    	mov	x8, #0x30               ; =48
100000d60: 9ac80800    	udiv	x0, x0, x8
100000d64: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d68: 910083ff    	add	sp, sp, #0x20
100000d6c: d65f03c0    	ret

0000000100000d70 <__ZSt28__throw_bad_array_new_lengthB8ne200100v>:
100000d70: d10083ff    	sub	sp, sp, #0x20
100000d74: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d78: 910043fd    	add	x29, sp, #0x10
100000d7c: d2800100    	mov	x0, #0x8                ; =8
100000d80: 94000389    	bl	0x100001ba4 <___stack_chk_guard+0x100001ba4>
100000d84: f90007e0    	str	x0, [sp, #0x8]
100000d88: 9400039c    	bl	0x100001bf8 <___stack_chk_guard+0x100001bf8>
100000d8c: f94007e0    	ldr	x0, [sp, #0x8]
100000d90: 90000021    	adrp	x1, 0x100004000 <___stack_chk_guard+0x100004000>
100000d94: f9404421    	ldr	x1, [x1, #0x88]
100000d98: 90000022    	adrp	x2, 0x100004000 <___stack_chk_guard+0x100004000>
100000d9c: f9404842    	ldr	x2, [x2, #0x90]
100000da0: 94000384    	bl	0x100001bb0 <___stack_chk_guard+0x100001bb0>

0000000100000da4 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPT_NS_15__element_countEm>:
100000da4: d10103ff    	sub	sp, sp, #0x40
100000da8: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000dac: 9100c3fd    	add	x29, sp, #0x30
100000db0: f81f03a0    	stur	x0, [x29, #-0x10]
100000db4: f9000fe1    	str	x1, [sp, #0x18]
100000db8: f85f03a8    	ldur	x8, [x29, #-0x10]
100000dbc: d2800609    	mov	x9, #0x30               ; =48
100000dc0: 9b097d08    	mul	x8, x8, x9
100000dc4: f9000be8    	str	x8, [sp, #0x10]
100000dc8: f9400fe0    	ldr	x0, [sp, #0x18]
100000dcc: 94000019    	bl	0x100000e30 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000dd0: 36000120    	tbz	w0, #0x0, 0x100000df4 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPT_NS_15__element_countEm+0x50>
100000dd4: 14000001    	b	0x100000dd8 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPT_NS_15__element_countEm+0x34>
100000dd8: f9400fe8    	ldr	x8, [sp, #0x18]
100000ddc: f90007e8    	str	x8, [sp, #0x8]
100000de0: f9400be0    	ldr	x0, [sp, #0x10]
100000de4: f94007e1    	ldr	x1, [sp, #0x8]
100000de8: 94000019    	bl	0x100000e4c <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEJmSt11align_val_tEEEPvDpT0_>
100000dec: f81f83a0    	stur	x0, [x29, #-0x8]
100000df0: 14000005    	b	0x100000e04 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPT_NS_15__element_countEm+0x60>
100000df4: f9400be0    	ldr	x0, [sp, #0x10]
100000df8: 94000020    	bl	0x100000e78 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPvm>
100000dfc: f81f83a0    	stur	x0, [x29, #-0x8]
100000e00: 14000001    	b	0x100000e04 <__ZNSt3__117__libcpp_allocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPT_NS_15__element_countEm+0x60>
100000e04: f85f83a0    	ldur	x0, [x29, #-0x8]
100000e08: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000e0c: 910103ff    	add	sp, sp, #0x40
100000e10: d65f03c0    	ret

0000000100000e14 <__ZNSt3__114numeric_limitsImE3maxB8ne200100Ev>:
100000e14: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000e18: 910003fd    	mov	x29, sp
100000e1c: 94000003    	bl	0x100000e28 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>
100000e20: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000e24: d65f03c0    	ret

0000000100000e28 <__ZNSt3__123__libcpp_numeric_limitsImLb1EE3maxB8ne200100Ev>:
100000e28: 92800000    	mov	x0, #-0x1               ; =-1
100000e2c: d65f03c0    	ret

0000000100000e30 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000e30: d10043ff    	sub	sp, sp, #0x10
100000e34: f90007e0    	str	x0, [sp, #0x8]
100000e38: f94007e8    	ldr	x8, [sp, #0x8]
100000e3c: f1004108    	subs	x8, x8, #0x10
100000e40: 1a9f97e0    	cset	w0, hi
100000e44: 910043ff    	add	sp, sp, #0x10
100000e48: d65f03c0    	ret

0000000100000e4c <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEJmSt11align_val_tEEEPvDpT0_>:
100000e4c: d10083ff    	sub	sp, sp, #0x20
100000e50: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e54: 910043fd    	add	x29, sp, #0x10
100000e58: f90007e0    	str	x0, [sp, #0x8]
100000e5c: f90003e1    	str	x1, [sp]
100000e60: f94007e0    	ldr	x0, [sp, #0x8]
100000e64: f94003e1    	ldr	x1, [sp]
100000e68: 94000367    	bl	0x100001c04 <___stack_chk_guard+0x100001c04>
100000e6c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e70: 910083ff    	add	sp, sp, #0x20
100000e74: d65f03c0    	ret

0000000100000e78 <__ZNSt3__121__libcpp_operator_newB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEPvm>:
100000e78: d10083ff    	sub	sp, sp, #0x20
100000e7c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000e80: 910043fd    	add	x29, sp, #0x10
100000e84: f90007e0    	str	x0, [sp, #0x8]
100000e88: f94007e0    	ldr	x0, [sp, #0x8]
100000e8c: 94000361    	bl	0x100001c10 <___stack_chk_guard+0x100001c10>
100000e90: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000e94: 910083ff    	add	sp, sp, #0x20
100000e98: d65f03c0    	ret

0000000100000e9c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_>:
100000e9c: d10103ff    	sub	sp, sp, #0x40
100000ea0: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000ea4: 9100c3fd    	add	x29, sp, #0x30
100000ea8: f81f03a0    	stur	x0, [x29, #-0x10]
100000eac: f9000fe1    	str	x1, [sp, #0x18]
100000eb0: f85f03a0    	ldur	x0, [x29, #-0x10]
100000eb4: f90003e0    	str	x0, [sp]
100000eb8: d2800001    	mov	x1, #0x0                ; =0
100000ebc: 94000027    	bl	0x100000f58 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000ec0: f94003e8    	ldr	x8, [sp]
100000ec4: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000ec8: 91040129    	add	x9, x9, #0x100
100000ecc: 91004129    	add	x9, x9, #0x10
100000ed0: f9000109    	str	x9, [x8]
100000ed4: 91006100    	add	x0, x8, #0x18
100000ed8: d10007a1    	sub	x1, x29, #0x1
100000edc: 94000032    	bl	0x100000fa4 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageC1B8ne200100EOS3_>
100000ee0: 14000001    	b	0x100000ee4 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_+0x48>
100000ee4: f94003e0    	ldr	x0, [sp]
100000ee8: 9400003c    	bl	0x100000fd8 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE11__get_allocB8ne200100Ev>
100000eec: f94003e0    	ldr	x0, [sp]
100000ef0: 97ffff30    	bl	0x100000bb0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE10__get_elemB8ne200100Ev>
100000ef4: aa0003e1    	mov	x1, x0
100000ef8: f9400fe2    	ldr	x2, [sp, #0x18]
100000efc: 91002fe0    	add	x0, sp, #0xb
100000f00: 94000347    	bl	0x100001c1c <___stack_chk_guard+0x100001c1c>
100000f04: 14000001    	b	0x100000f08 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_+0x6c>
100000f08: f94003e0    	ldr	x0, [sp]
100000f0c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000f10: 910103ff    	add	sp, sp, #0x40
100000f14: d65f03c0    	ret
100000f18: f9000be0    	str	x0, [sp, #0x10]
100000f1c: aa0103e8    	mov	x8, x1
100000f20: b9000fe8    	str	w8, [sp, #0xc]
100000f24: 14000008    	b	0x100000f44 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_+0xa8>
100000f28: f94003e8    	ldr	x8, [sp]
100000f2c: f9000be0    	str	x0, [sp, #0x10]
100000f30: aa0103e9    	mov	x9, x1
100000f34: b9000fe9    	str	w9, [sp, #0xc]
100000f38: 91006100    	add	x0, x8, #0x18
100000f3c: 9400003d    	bl	0x100001030 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageD1B8ne200100Ev>
100000f40: 14000001    	b	0x100000f44 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_+0xa8>
100000f44: f94003e0    	ldr	x0, [sp]
100000f48: 94000338    	bl	0x100001c28 <___stack_chk_guard+0x100001c28>
100000f4c: 14000001    	b	0x100000f50 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEEC2B8ne200100IJiES3_Li0EEES3_DpOT_+0xb4>
100000f50: f9400be0    	ldr	x0, [sp, #0x10]
100000f54: 9400030e    	bl	0x100001b8c <___stack_chk_guard+0x100001b8c>

0000000100000f58 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
100000f58: d100c3ff    	sub	sp, sp, #0x30
100000f5c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000f60: 910083fd    	add	x29, sp, #0x20
100000f64: f81f83a0    	stur	x0, [x29, #-0x8]
100000f68: f9000be1    	str	x1, [sp, #0x10]
100000f6c: f85f83a0    	ldur	x0, [x29, #-0x8]
100000f70: f90007e0    	str	x0, [sp, #0x8]
100000f74: f9400be1    	ldr	x1, [sp, #0x10]
100000f78: 94000070    	bl	0x100001138 <__ZNSt3__114__shared_countC2B8ne200100El>
100000f7c: f94007e0    	ldr	x0, [sp, #0x8]
100000f80: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000f84: f9405d08    	ldr	x8, [x8, #0xb8]
100000f88: 91004108    	add	x8, x8, #0x10
100000f8c: f9000008    	str	x8, [x0]
100000f90: f9400be8    	ldr	x8, [sp, #0x10]
100000f94: f9000808    	str	x8, [x0, #0x10]
100000f98: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000f9c: 9100c3ff    	add	sp, sp, #0x30
100000fa0: d65f03c0    	ret

0000000100000fa4 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageC1B8ne200100EOS3_>:
100000fa4: d100c3ff    	sub	sp, sp, #0x30
100000fa8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000fac: 910083fd    	add	x29, sp, #0x20
100000fb0: f81f83a0    	stur	x0, [x29, #-0x8]
100000fb4: f9000be1    	str	x1, [sp, #0x10]
100000fb8: f85f83a0    	ldur	x0, [x29, #-0x8]
100000fbc: f90007e0    	str	x0, [sp, #0x8]
100000fc0: f9400be1    	ldr	x1, [sp, #0x10]
100000fc4: 94000069    	bl	0x100001168 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageC2B8ne200100EOS3_>
100000fc8: f94007e0    	ldr	x0, [sp, #0x8]
100000fcc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000fd0: 9100c3ff    	add	sp, sp, #0x30
100000fd4: d65f03c0    	ret

0000000100000fd8 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE11__get_allocB8ne200100Ev>:
100000fd8: d10083ff    	sub	sp, sp, #0x20
100000fdc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000fe0: 910043fd    	add	x29, sp, #0x10
100000fe4: f90007e0    	str	x0, [sp, #0x8]
100000fe8: f94007e8    	ldr	x8, [sp, #0x8]
100000fec: 91006100    	add	x0, x8, #0x18
100000ff0: 9400006a    	bl	0x100001198 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_Storage11__get_allocB8ne200100Ev>
100000ff4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000ff8: 910083ff    	add	sp, sp, #0x20
100000ffc: d65f03c0    	ret

0000000100001000 <__ZNSt3__116allocator_traitsINS_9allocatorI4NodeEEE9constructB8ne200100IS2_JiEvLi0EEEvRS3_PT_DpOT0_>:
100001000: d100c3ff    	sub	sp, sp, #0x30
100001004: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001008: 910083fd    	add	x29, sp, #0x20
10000100c: f81f83a0    	stur	x0, [x29, #-0x8]
100001010: f9000be1    	str	x1, [sp, #0x10]
100001014: f90007e2    	str	x2, [sp, #0x8]
100001018: f9400be0    	ldr	x0, [sp, #0x10]
10000101c: f94007e1    	ldr	x1, [sp, #0x8]
100001020: 94000063    	bl	0x1000011ac <__ZNSt3__114__construct_atB8ne200100I4NodeJiEPS1_EEPT_S4_DpOT0_>
100001024: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001028: 9100c3ff    	add	sp, sp, #0x30
10000102c: d65f03c0    	ret

0000000100001030 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageD1B8ne200100Ev>:
100001030: d10083ff    	sub	sp, sp, #0x20
100001034: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001038: 910043fd    	add	x29, sp, #0x10
10000103c: f90007e0    	str	x0, [sp, #0x8]
100001040: f94007e0    	ldr	x0, [sp, #0x8]
100001044: f90003e0    	str	x0, [sp]
100001048: 940000aa    	bl	0x1000012f0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageD2B8ne200100Ev>
10000104c: f94003e0    	ldr	x0, [sp]
100001050: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001054: 910083ff    	add	sp, sp, #0x20
100001058: d65f03c0    	ret

000000010000105c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED1Ev>:
10000105c: d10083ff    	sub	sp, sp, #0x20
100001060: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001064: 910043fd    	add	x29, sp, #0x10
100001068: f90007e0    	str	x0, [sp, #0x8]
10000106c: f94007e0    	ldr	x0, [sp, #0x8]
100001070: f90003e0    	str	x0, [sp]
100001074: 940000aa    	bl	0x10000131c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED2Ev>
100001078: f94003e0    	ldr	x0, [sp]
10000107c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001080: 910083ff    	add	sp, sp, #0x20
100001084: d65f03c0    	ret

0000000100001088 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED0Ev>:
100001088: d10083ff    	sub	sp, sp, #0x20
10000108c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001090: 910043fd    	add	x29, sp, #0x10
100001094: f90007e0    	str	x0, [sp, #0x8]
100001098: f94007e0    	ldr	x0, [sp, #0x8]
10000109c: f90003e0    	str	x0, [sp]
1000010a0: 97ffffef    	bl	0x10000105c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED1Ev>
1000010a4: f94003e0    	ldr	x0, [sp]
1000010a8: 940002e3    	bl	0x100001c34 <___stack_chk_guard+0x100001c34>
1000010ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000010b0: 910083ff    	add	sp, sp, #0x20
1000010b4: d65f03c0    	ret

00000001000010b8 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
1000010b8: d10083ff    	sub	sp, sp, #0x20
1000010bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000010c0: 910043fd    	add	x29, sp, #0x10
1000010c4: f90007e0    	str	x0, [sp, #0x8]
1000010c8: f94007e0    	ldr	x0, [sp, #0x8]
1000010cc: 940002dd    	bl	0x100001c40 <___stack_chk_guard+0x100001c40>
1000010d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000010d4: 910083ff    	add	sp, sp, #0x20
1000010d8: d65f03c0    	ret

00000001000010dc <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
1000010dc: d100c3ff    	sub	sp, sp, #0x30
1000010e0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000010e4: 910083fd    	add	x29, sp, #0x20
1000010e8: f81f83a0    	stur	x0, [x29, #-0x8]
1000010ec: f85f83a0    	ldur	x0, [x29, #-0x8]
1000010f0: f90003e0    	str	x0, [sp]
1000010f4: 97ffffb9    	bl	0x100000fd8 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE11__get_allocB8ne200100Ev>
1000010f8: aa0003e1    	mov	x1, x0
1000010fc: d10027a0    	sub	x0, x29, #0x9
100001100: f90007e0    	str	x0, [sp, #0x8]
100001104: 97fffed4    	bl	0x100000c54 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEEC1B8ne200100IS2_EERKNS0_IT_EE>
100001108: f94003e8    	ldr	x8, [sp]
10000110c: 91006100    	add	x0, x8, #0x18
100001110: 97ffffc8    	bl	0x100001030 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageD1B8ne200100Ev>
100001114: f94003e0    	ldr	x0, [sp]
100001118: 94000107    	bl	0x100001534 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEE10pointer_toB8ne200100ERS5_>
10000111c: aa0003e1    	mov	x1, x0
100001120: f94007e0    	ldr	x0, [sp, #0x8]
100001124: d2800022    	mov	x2, #0x1                ; =1
100001128: 940000f6    	bl	0x100001500 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE10deallocateB8ne200100ERS6_PS5_m>
10000112c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001130: 9100c3ff    	add	sp, sp, #0x30
100001134: d65f03c0    	ret

0000000100001138 <__ZNSt3__114__shared_countC2B8ne200100El>:
100001138: d10043ff    	sub	sp, sp, #0x10
10000113c: f90007e0    	str	x0, [sp, #0x8]
100001140: f90003e1    	str	x1, [sp]
100001144: f94007e0    	ldr	x0, [sp, #0x8]
100001148: f0000008    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
10000114c: f9406908    	ldr	x8, [x8, #0xd0]
100001150: 91004108    	add	x8, x8, #0x10
100001154: f9000008    	str	x8, [x0]
100001158: f94003e8    	ldr	x8, [sp]
10000115c: f9000408    	str	x8, [x0, #0x8]
100001160: 910043ff    	add	sp, sp, #0x10
100001164: d65f03c0    	ret

0000000100001168 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageC2B8ne200100EOS3_>:
100001168: d100c3ff    	sub	sp, sp, #0x30
10000116c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001170: 910083fd    	add	x29, sp, #0x20
100001174: f81f83a0    	stur	x0, [x29, #-0x8]
100001178: f9000be1    	str	x1, [sp, #0x10]
10000117c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001180: f90007e0    	str	x0, [sp, #0x8]
100001184: 94000005    	bl	0x100001198 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_Storage11__get_allocB8ne200100Ev>
100001188: f94007e0    	ldr	x0, [sp, #0x8]
10000118c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001190: 9100c3ff    	add	sp, sp, #0x30
100001194: d65f03c0    	ret

0000000100001198 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_Storage11__get_allocB8ne200100Ev>:
100001198: d10043ff    	sub	sp, sp, #0x10
10000119c: f90007e0    	str	x0, [sp, #0x8]
1000011a0: f94007e0    	ldr	x0, [sp, #0x8]
1000011a4: 910043ff    	add	sp, sp, #0x10
1000011a8: d65f03c0    	ret

00000001000011ac <__ZNSt3__114__construct_atB8ne200100I4NodeJiEPS1_EEPT_S4_DpOT0_>:
1000011ac: d10083ff    	sub	sp, sp, #0x20
1000011b0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000011b4: 910043fd    	add	x29, sp, #0x10
1000011b8: f90007e0    	str	x0, [sp, #0x8]
1000011bc: f90003e1    	str	x1, [sp]
1000011c0: f94007e0    	ldr	x0, [sp, #0x8]
1000011c4: f94003e1    	ldr	x1, [sp]
1000011c8: 94000004    	bl	0x1000011d8 <__ZNSt3__112construct_atB8ne200100I4NodeJiEPS1_EEPT_S4_DpOT0_>
1000011cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000011d0: 910083ff    	add	sp, sp, #0x20
1000011d4: d65f03c0    	ret

00000001000011d8 <__ZNSt3__112construct_atB8ne200100I4NodeJiEPS1_EEPT_S4_DpOT0_>:
1000011d8: d100c3ff    	sub	sp, sp, #0x30
1000011dc: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000011e0: 910083fd    	add	x29, sp, #0x20
1000011e4: f81f83a0    	stur	x0, [x29, #-0x8]
1000011e8: f9000be1    	str	x1, [sp, #0x10]
1000011ec: f85f83a0    	ldur	x0, [x29, #-0x8]
1000011f0: f90007e0    	str	x0, [sp, #0x8]
1000011f4: f9400be8    	ldr	x8, [sp, #0x10]
1000011f8: b9400101    	ldr	w1, [x8]
1000011fc: 94000005    	bl	0x100001210 <__ZN4NodeC1Ei>
100001200: f94007e0    	ldr	x0, [sp, #0x8]
100001204: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001208: 9100c3ff    	add	sp, sp, #0x30
10000120c: d65f03c0    	ret

0000000100001210 <__ZN4NodeC1Ei>:
100001210: d100c3ff    	sub	sp, sp, #0x30
100001214: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001218: 910083fd    	add	x29, sp, #0x20
10000121c: f81f83a0    	stur	x0, [x29, #-0x8]
100001220: b81f43a1    	stur	w1, [x29, #-0xc]
100001224: f85f83a0    	ldur	x0, [x29, #-0x8]
100001228: f90007e0    	str	x0, [sp, #0x8]
10000122c: b85f43a1    	ldur	w1, [x29, #-0xc]
100001230: 94000005    	bl	0x100001244 <__ZN4NodeC2Ei>
100001234: f94007e0    	ldr	x0, [sp, #0x8]
100001238: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000123c: 9100c3ff    	add	sp, sp, #0x30
100001240: d65f03c0    	ret

0000000100001244 <__ZN4NodeC2Ei>:
100001244: d100c3ff    	sub	sp, sp, #0x30
100001248: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000124c: 910083fd    	add	x29, sp, #0x20
100001250: f81f83a0    	stur	x0, [x29, #-0x8]
100001254: b81f43a1    	stur	w1, [x29, #-0xc]
100001258: f85f83a0    	ldur	x0, [x29, #-0x8]
10000125c: f90007e0    	str	x0, [sp, #0x8]
100001260: 94000007    	bl	0x10000127c <__ZNSt3__123enable_shared_from_thisI4NodeEC2B8ne200100Ev>
100001264: f94007e0    	ldr	x0, [sp, #0x8]
100001268: b85f43a8    	ldur	w8, [x29, #-0xc]
10000126c: b9001008    	str	w8, [x0, #0x10]
100001270: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001274: 9100c3ff    	add	sp, sp, #0x30
100001278: d65f03c0    	ret

000000010000127c <__ZNSt3__123enable_shared_from_thisI4NodeEC2B8ne200100Ev>:
10000127c: d10083ff    	sub	sp, sp, #0x20
100001280: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001284: 910043fd    	add	x29, sp, #0x10
100001288: f90007e0    	str	x0, [sp, #0x8]
10000128c: f94007e0    	ldr	x0, [sp, #0x8]
100001290: f90003e0    	str	x0, [sp]
100001294: 94000005    	bl	0x1000012a8 <__ZNSt3__18weak_ptrI4NodeEC1Ev>
100001298: f94003e0    	ldr	x0, [sp]
10000129c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012a0: 910083ff    	add	sp, sp, #0x20
1000012a4: d65f03c0    	ret

00000001000012a8 <__ZNSt3__18weak_ptrI4NodeEC1Ev>:
1000012a8: d10083ff    	sub	sp, sp, #0x20
1000012ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012b0: 910043fd    	add	x29, sp, #0x10
1000012b4: f90007e0    	str	x0, [sp, #0x8]
1000012b8: f94007e0    	ldr	x0, [sp, #0x8]
1000012bc: f90003e0    	str	x0, [sp]
1000012c0: 94000005    	bl	0x1000012d4 <__ZNSt3__18weak_ptrI4NodeEC2Ev>
1000012c4: f94003e0    	ldr	x0, [sp]
1000012c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000012cc: 910083ff    	add	sp, sp, #0x20
1000012d0: d65f03c0    	ret

00000001000012d4 <__ZNSt3__18weak_ptrI4NodeEC2Ev>:
1000012d4: d10043ff    	sub	sp, sp, #0x10
1000012d8: f90007e0    	str	x0, [sp, #0x8]
1000012dc: f94007e0    	ldr	x0, [sp, #0x8]
1000012e0: f900001f    	str	xzr, [x0]
1000012e4: f900041f    	str	xzr, [x0, #0x8]
1000012e8: 910043ff    	add	sp, sp, #0x10
1000012ec: d65f03c0    	ret

00000001000012f0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageD2B8ne200100Ev>:
1000012f0: d10083ff    	sub	sp, sp, #0x20
1000012f4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000012f8: 910043fd    	add	x29, sp, #0x10
1000012fc: f90007e0    	str	x0, [sp, #0x8]
100001300: f94007e0    	ldr	x0, [sp, #0x8]
100001304: f90003e0    	str	x0, [sp]
100001308: 97ffffa4    	bl	0x100001198 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_Storage11__get_allocB8ne200100Ev>
10000130c: f94003e0    	ldr	x0, [sp]
100001310: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001314: 910083ff    	add	sp, sp, #0x20
100001318: d65f03c0    	ret

000000010000131c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED2Ev>:
10000131c: d10083ff    	sub	sp, sp, #0x20
100001320: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001324: 910043fd    	add	x29, sp, #0x10
100001328: f90007e0    	str	x0, [sp, #0x8]
10000132c: f94007e8    	ldr	x8, [sp, #0x8]
100001330: f90003e8    	str	x8, [sp]
100001334: f0000009    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100001338: 91040129    	add	x9, x9, #0x100
10000133c: 91004129    	add	x9, x9, #0x10
100001340: f9000109    	str	x9, [x8]
100001344: 91006100    	add	x0, x8, #0x18
100001348: 97ffff3a    	bl	0x100001030 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_StorageD1B8ne200100Ev>
10000134c: f94003e0    	ldr	x0, [sp]
100001350: 94000236    	bl	0x100001c28 <___stack_chk_guard+0x100001c28>
100001354: f94003e0    	ldr	x0, [sp]
100001358: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000135c: 910083ff    	add	sp, sp, #0x20
100001360: d65f03c0    	ret

0000000100001364 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE21__on_zero_shared_implB8ne200100IS3_Li0EEEvv>:
100001364: d100c3ff    	sub	sp, sp, #0x30
100001368: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000136c: 910083fd    	add	x29, sp, #0x20
100001370: f81f83a0    	stur	x0, [x29, #-0x8]
100001374: f85f83a0    	ldur	x0, [x29, #-0x8]
100001378: f90007e0    	str	x0, [sp, #0x8]
10000137c: 97ffff17    	bl	0x100000fd8 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE11__get_allocB8ne200100Ev>
100001380: f94007e0    	ldr	x0, [sp, #0x8]
100001384: 97fffe0b    	bl	0x100000bb0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE10__get_elemB8ne200100Ev>
100001388: aa0003e1    	mov	x1, x0
10000138c: d10027a0    	sub	x0, x29, #0x9
100001390: 9400022f    	bl	0x100001c4c <___stack_chk_guard+0x100001c4c>
100001394: 14000001    	b	0x100001398 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE21__on_zero_shared_implB8ne200100IS3_Li0EEEvv+0x34>
100001398: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000139c: 9100c3ff    	add	sp, sp, #0x30
1000013a0: d65f03c0    	ret
1000013a4: 9400000b    	bl	0x1000013d0 <___clang_call_terminate>

00000001000013a8 <__ZNSt3__116allocator_traitsINS_9allocatorI4NodeEEE7destroyB8ne200100IS2_vLi0EEEvRS3_PT_>:
1000013a8: d10083ff    	sub	sp, sp, #0x20
1000013ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000013b0: 910043fd    	add	x29, sp, #0x10
1000013b4: f90007e0    	str	x0, [sp, #0x8]
1000013b8: f90003e1    	str	x1, [sp]
1000013bc: f94003e0    	ldr	x0, [sp]
1000013c0: 94000008    	bl	0x1000013e0 <__ZNSt3__112__destroy_atB8ne200100I4NodeLi0EEEvPT_>
1000013c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000013c8: 910083ff    	add	sp, sp, #0x20
1000013cc: d65f03c0    	ret

00000001000013d0 <___clang_call_terminate>:
1000013d0: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000013d4: 910003fd    	mov	x29, sp
1000013d8: 94000220    	bl	0x100001c58 <___stack_chk_guard+0x100001c58>
1000013dc: 94000222    	bl	0x100001c64 <___stack_chk_guard+0x100001c64>

00000001000013e0 <__ZNSt3__112__destroy_atB8ne200100I4NodeLi0EEEvPT_>:
1000013e0: d10083ff    	sub	sp, sp, #0x20
1000013e4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000013e8: 910043fd    	add	x29, sp, #0x10
1000013ec: f90007e0    	str	x0, [sp, #0x8]
1000013f0: f94007e0    	ldr	x0, [sp, #0x8]
1000013f4: 94000004    	bl	0x100001404 <__ZN4NodeD1Ev>
1000013f8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000013fc: 910083ff    	add	sp, sp, #0x20
100001400: d65f03c0    	ret

0000000100001404 <__ZN4NodeD1Ev>:
100001404: d10083ff    	sub	sp, sp, #0x20
100001408: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000140c: 910043fd    	add	x29, sp, #0x10
100001410: f90007e0    	str	x0, [sp, #0x8]
100001414: f94007e0    	ldr	x0, [sp, #0x8]
100001418: f90003e0    	str	x0, [sp]
10000141c: 94000005    	bl	0x100001430 <__ZN4NodeD2Ev>
100001420: f94003e0    	ldr	x0, [sp]
100001424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001428: 910083ff    	add	sp, sp, #0x20
10000142c: d65f03c0    	ret

0000000100001430 <__ZN4NodeD2Ev>:
100001430: d10083ff    	sub	sp, sp, #0x20
100001434: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001438: 910043fd    	add	x29, sp, #0x10
10000143c: f90007e0    	str	x0, [sp, #0x8]
100001440: f94007e0    	ldr	x0, [sp, #0x8]
100001444: f90003e0    	str	x0, [sp]
100001448: 94000005    	bl	0x10000145c <__ZNSt3__123enable_shared_from_thisI4NodeED2B8ne200100Ev>
10000144c: f94003e0    	ldr	x0, [sp]
100001450: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001454: 910083ff    	add	sp, sp, #0x20
100001458: d65f03c0    	ret

000000010000145c <__ZNSt3__123enable_shared_from_thisI4NodeED2B8ne200100Ev>:
10000145c: d10083ff    	sub	sp, sp, #0x20
100001460: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001464: 910043fd    	add	x29, sp, #0x10
100001468: f90007e0    	str	x0, [sp, #0x8]
10000146c: f94007e0    	ldr	x0, [sp, #0x8]
100001470: f90003e0    	str	x0, [sp]
100001474: 94000005    	bl	0x100001488 <__ZNSt3__18weak_ptrI4NodeED1Ev>
100001478: f94003e0    	ldr	x0, [sp]
10000147c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001480: 910083ff    	add	sp, sp, #0x20
100001484: d65f03c0    	ret

0000000100001488 <__ZNSt3__18weak_ptrI4NodeED1Ev>:
100001488: d10083ff    	sub	sp, sp, #0x20
10000148c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001490: 910043fd    	add	x29, sp, #0x10
100001494: f90007e0    	str	x0, [sp, #0x8]
100001498: f94007e0    	ldr	x0, [sp, #0x8]
10000149c: f90003e0    	str	x0, [sp]
1000014a0: 94000005    	bl	0x1000014b4 <__ZNSt3__18weak_ptrI4NodeED2Ev>
1000014a4: f94003e0    	ldr	x0, [sp]
1000014a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000014ac: 910083ff    	add	sp, sp, #0x20
1000014b0: d65f03c0    	ret

00000001000014b4 <__ZNSt3__18weak_ptrI4NodeED2Ev>:
1000014b4: d100c3ff    	sub	sp, sp, #0x30
1000014b8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000014bc: 910083fd    	add	x29, sp, #0x20
1000014c0: f9000be0    	str	x0, [sp, #0x10]
1000014c4: f9400be8    	ldr	x8, [sp, #0x10]
1000014c8: f90007e8    	str	x8, [sp, #0x8]
1000014cc: aa0803e9    	mov	x9, x8
1000014d0: f81f83a9    	stur	x9, [x29, #-0x8]
1000014d4: f9400508    	ldr	x8, [x8, #0x8]
1000014d8: b40000c8    	cbz	x8, 0x1000014f0 <__ZNSt3__18weak_ptrI4NodeED2Ev+0x3c>
1000014dc: 14000001    	b	0x1000014e0 <__ZNSt3__18weak_ptrI4NodeED2Ev+0x2c>
1000014e0: f94007e8    	ldr	x8, [sp, #0x8]
1000014e4: f9400500    	ldr	x0, [x8, #0x8]
1000014e8: 940001b5    	bl	0x100001bbc <___stack_chk_guard+0x100001bbc>
1000014ec: 14000001    	b	0x1000014f0 <__ZNSt3__18weak_ptrI4NodeED2Ev+0x3c>
1000014f0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000014f4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000014f8: 9100c3ff    	add	sp, sp, #0x30
1000014fc: d65f03c0    	ret

0000000100001500 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE10deallocateB8ne200100ERS6_PS5_m>:
100001500: d100c3ff    	sub	sp, sp, #0x30
100001504: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001508: 910083fd    	add	x29, sp, #0x20
10000150c: f81f83a0    	stur	x0, [x29, #-0x8]
100001510: f9000be1    	str	x1, [sp, #0x10]
100001514: f90007e2    	str	x2, [sp, #0x8]
100001518: f85f83a0    	ldur	x0, [x29, #-0x8]
10000151c: f9400be1    	ldr	x1, [sp, #0x10]
100001520: f94007e2    	ldr	x2, [sp, #0x8]
100001524: 94000009    	bl	0x100001548 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEE10deallocateB8ne200100EPS4_m>
100001528: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000152c: 9100c3ff    	add	sp, sp, #0x30
100001530: d65f03c0    	ret

0000000100001534 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEE10pointer_toB8ne200100ERS5_>:
100001534: d10043ff    	sub	sp, sp, #0x10
100001538: f90007e0    	str	x0, [sp, #0x8]
10000153c: f94007e0    	ldr	x0, [sp, #0x8]
100001540: 910043ff    	add	sp, sp, #0x10
100001544: d65f03c0    	ret

0000000100001548 <__ZNSt3__19allocatorINS_20__shared_ptr_emplaceI4NodeNS0_IS2_EEEEE10deallocateB8ne200100EPS4_m>:
100001548: d100c3ff    	sub	sp, sp, #0x30
10000154c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001550: 910083fd    	add	x29, sp, #0x20
100001554: f81f83a0    	stur	x0, [x29, #-0x8]
100001558: f9000be1    	str	x1, [sp, #0x10]
10000155c: f90007e2    	str	x2, [sp, #0x8]
100001560: f9400be0    	ldr	x0, [sp, #0x10]
100001564: f94007e1    	ldr	x1, [sp, #0x8]
100001568: d2800102    	mov	x2, #0x8                ; =8
10000156c: 94000004    	bl	0x10000157c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
100001570: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001574: 9100c3ff    	add	sp, sp, #0x30
100001578: d65f03c0    	ret

000000010000157c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
10000157c: d10103ff    	sub	sp, sp, #0x40
100001580: a9037bfd    	stp	x29, x30, [sp, #0x30]
100001584: 9100c3fd    	add	x29, sp, #0x30
100001588: f81f83a0    	stur	x0, [x29, #-0x8]
10000158c: f81f03a1    	stur	x1, [x29, #-0x10]
100001590: f9000fe2    	str	x2, [sp, #0x18]
100001594: f85f03a8    	ldur	x8, [x29, #-0x10]
100001598: d2800609    	mov	x9, #0x30               ; =48
10000159c: 9b097d08    	mul	x8, x8, x9
1000015a0: f9000be8    	str	x8, [sp, #0x10]
1000015a4: f9400fe0    	ldr	x0, [sp, #0x18]
1000015a8: 97fffe22    	bl	0x100000e30 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
1000015ac: 36000100    	tbz	w0, #0x0, 0x1000015cc <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x50>
1000015b0: 14000001    	b	0x1000015b4 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x38>
1000015b4: f9400fe8    	ldr	x8, [sp, #0x18]
1000015b8: f90007e8    	str	x8, [sp, #0x8]
1000015bc: f85f83a0    	ldur	x0, [x29, #-0x8]
1000015c0: f94007e1    	ldr	x1, [sp, #0x8]
1000015c4: 94000008    	bl	0x1000015e4 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEESt11align_val_tEEEvDpT_>
1000015c8: 14000004    	b	0x1000015d8 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x5c>
1000015cc: f85f83a0    	ldur	x0, [x29, #-0x8]
1000015d0: 94000010    	bl	0x100001610 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEEvDpT_>
1000015d4: 14000001    	b	0x1000015d8 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x5c>
1000015d8: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000015dc: 910103ff    	add	sp, sp, #0x40
1000015e0: d65f03c0    	ret

00000001000015e4 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEESt11align_val_tEEEvDpT_>:
1000015e4: d10083ff    	sub	sp, sp, #0x20
1000015e8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000015ec: 910043fd    	add	x29, sp, #0x10
1000015f0: f90007e0    	str	x0, [sp, #0x8]
1000015f4: f90003e1    	str	x1, [sp]
1000015f8: f94007e0    	ldr	x0, [sp, #0x8]
1000015fc: f94003e1    	ldr	x1, [sp]
100001600: 9400019c    	bl	0x100001c70 <___stack_chk_guard+0x100001c70>
100001604: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001608: 910083ff    	add	sp, sp, #0x20
10000160c: d65f03c0    	ret

0000000100001610 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_emplaceI4NodeNS_9allocatorIS2_EEEEEEEvDpT_>:
100001610: d10083ff    	sub	sp, sp, #0x20
100001614: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001618: 910043fd    	add	x29, sp, #0x10
10000161c: f90007e0    	str	x0, [sp, #0x8]
100001620: f94007e0    	ldr	x0, [sp, #0x8]
100001624: 94000184    	bl	0x100001c34 <___stack_chk_guard+0x100001c34>
100001628: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000162c: 910083ff    	add	sp, sp, #0x20
100001630: d65f03c0    	ret

0000000100001634 <__ZNSt3__110shared_ptrI4NodeEC1B8ne200100Ev>:
100001634: d10083ff    	sub	sp, sp, #0x20
100001638: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000163c: 910043fd    	add	x29, sp, #0x10
100001640: f90007e0    	str	x0, [sp, #0x8]
100001644: f94007e0    	ldr	x0, [sp, #0x8]
100001648: f90003e0    	str	x0, [sp]
10000164c: 94000022    	bl	0x1000016d4 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100Ev>
100001650: f94003e0    	ldr	x0, [sp]
100001654: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001658: 910083ff    	add	sp, sp, #0x20
10000165c: d65f03c0    	ret

0000000100001660 <__ZNSt3__110shared_ptrI4NodeE18__enable_weak_thisB8ne200100IS1_S1_Li0EEEvPKNS_23enable_shared_from_thisIT_EEPT0_>:
100001660: d10143ff    	sub	sp, sp, #0x50
100001664: a9047bfd    	stp	x29, x30, [sp, #0x40]
100001668: 910103fd    	add	x29, sp, #0x40
10000166c: f81f83a0    	stur	x0, [x29, #-0x8]
100001670: f81f03a1    	stur	x1, [x29, #-0x10]
100001674: f81e83a2    	stur	x2, [x29, #-0x18]
100001678: f85f83a8    	ldur	x8, [x29, #-0x8]
10000167c: f9000be8    	str	x8, [sp, #0x10]
100001680: f85f03a8    	ldur	x8, [x29, #-0x10]
100001684: b4000228    	cbz	x8, 0x1000016c8 <__ZNSt3__110shared_ptrI4NodeE18__enable_weak_thisB8ne200100IS1_S1_Li0EEEvPKNS_23enable_shared_from_thisIT_EEPT0_+0x68>
100001688: 14000001    	b	0x10000168c <__ZNSt3__110shared_ptrI4NodeE18__enable_weak_thisB8ne200100IS1_S1_Li0EEEvPKNS_23enable_shared_from_thisIT_EEPT0_+0x2c>
10000168c: f85f03a0    	ldur	x0, [x29, #-0x10]
100001690: 94000018    	bl	0x1000016f0 <__ZNKSt3__18weak_ptrI4NodeE7expiredB8ne200100Ev>
100001694: 360001a0    	tbz	w0, #0x0, 0x1000016c8 <__ZNSt3__110shared_ptrI4NodeE18__enable_weak_thisB8ne200100IS1_S1_Li0EEEvPKNS_23enable_shared_from_thisIT_EEPT0_+0x68>
100001698: 14000001    	b	0x10000169c <__ZNSt3__110shared_ptrI4NodeE18__enable_weak_thisB8ne200100IS1_S1_Li0EEEvPKNS_23enable_shared_from_thisIT_EEPT0_+0x3c>
10000169c: f9400be1    	ldr	x1, [sp, #0x10]
1000016a0: f85e83a2    	ldur	x2, [x29, #-0x18]
1000016a4: 910063e0    	add	x0, sp, #0x18
1000016a8: f90007e0    	str	x0, [sp, #0x8]
1000016ac: 94000028    	bl	0x10000174c <__ZNSt3__110shared_ptrI4NodeEC1B8ne200100IS1_EERKNS0_IT_EEPS1_>
1000016b0: f94007e1    	ldr	x1, [sp, #0x8]
1000016b4: f85f03a0    	ldur	x0, [x29, #-0x10]
1000016b8: 94000171    	bl	0x100001c7c <___stack_chk_guard+0x100001c7c>
1000016bc: f94007e0    	ldr	x0, [sp, #0x8]
1000016c0: 97fffbe4    	bl	0x100000650 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
1000016c4: 14000001    	b	0x1000016c8 <__ZNSt3__110shared_ptrI4NodeE18__enable_weak_thisB8ne200100IS1_S1_Li0EEEvPKNS_23enable_shared_from_thisIT_EEPT0_+0x68>
1000016c8: a9447bfd    	ldp	x29, x30, [sp, #0x40]
1000016cc: 910143ff    	add	sp, sp, #0x50
1000016d0: d65f03c0    	ret

00000001000016d4 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100Ev>:
1000016d4: d10043ff    	sub	sp, sp, #0x10
1000016d8: f90007e0    	str	x0, [sp, #0x8]
1000016dc: f94007e0    	ldr	x0, [sp, #0x8]
1000016e0: f900001f    	str	xzr, [x0]
1000016e4: f900041f    	str	xzr, [x0, #0x8]
1000016e8: 910043ff    	add	sp, sp, #0x10
1000016ec: d65f03c0    	ret

00000001000016f0 <__ZNKSt3__18weak_ptrI4NodeE7expiredB8ne200100Ev>:
1000016f0: d100c3ff    	sub	sp, sp, #0x30
1000016f4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000016f8: 910083fd    	add	x29, sp, #0x20
1000016fc: f81f83a0    	stur	x0, [x29, #-0x8]
100001700: f85f83a8    	ldur	x8, [x29, #-0x8]
100001704: f90007e8    	str	x8, [sp, #0x8]
100001708: f9400508    	ldr	x8, [x8, #0x8]
10000170c: 52800029    	mov	w9, #0x1                ; =1
100001710: b81f43a9    	stur	w9, [x29, #-0xc]
100001714: b4000128    	cbz	x8, 0x100001738 <__ZNKSt3__18weak_ptrI4NodeE7expiredB8ne200100Ev+0x48>
100001718: 14000001    	b	0x10000171c <__ZNKSt3__18weak_ptrI4NodeE7expiredB8ne200100Ev+0x2c>
10000171c: f94007e8    	ldr	x8, [sp, #0x8]
100001720: f9400500    	ldr	x0, [x8, #0x8]
100001724: 9400002d    	bl	0x1000017d8 <__ZNKSt3__119__shared_weak_count9use_countB8ne200100Ev>
100001728: f1000008    	subs	x8, x0, #0x0
10000172c: 1a9f17e8    	cset	w8, eq
100001730: b81f43a8    	stur	w8, [x29, #-0xc]
100001734: 14000001    	b	0x100001738 <__ZNKSt3__18weak_ptrI4NodeE7expiredB8ne200100Ev+0x48>
100001738: b85f43a8    	ldur	w8, [x29, #-0xc]
10000173c: 12000100    	and	w0, w8, #0x1
100001740: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001744: 9100c3ff    	add	sp, sp, #0x30
100001748: d65f03c0    	ret

000000010000174c <__ZNSt3__110shared_ptrI4NodeEC1B8ne200100IS1_EERKNS0_IT_EEPS1_>:
10000174c: d100c3ff    	sub	sp, sp, #0x30
100001750: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001754: 910083fd    	add	x29, sp, #0x20
100001758: f81f83a0    	stur	x0, [x29, #-0x8]
10000175c: f9000be1    	str	x1, [sp, #0x10]
100001760: f90007e2    	str	x2, [sp, #0x8]
100001764: f85f83a0    	ldur	x0, [x29, #-0x8]
100001768: f90003e0    	str	x0, [sp]
10000176c: f9400be1    	ldr	x1, [sp, #0x10]
100001770: f94007e2    	ldr	x2, [sp, #0x8]
100001774: 94000039    	bl	0x100001858 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_EERKNS0_IT_EEPS1_>
100001778: f94003e0    	ldr	x0, [sp]
10000177c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001780: 9100c3ff    	add	sp, sp, #0x30
100001784: d65f03c0    	ret

0000000100001788 <__ZNSt3__18weak_ptrI4NodeEaSIS1_Li0EEERS2_RKNS_10shared_ptrIT_EE>:
100001788: d10103ff    	sub	sp, sp, #0x40
10000178c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100001790: 9100c3fd    	add	x29, sp, #0x30
100001794: f81f83a0    	stur	x0, [x29, #-0x8]
100001798: f81f03a1    	stur	x1, [x29, #-0x10]
10000179c: f85f83a8    	ldur	x8, [x29, #-0x8]
1000017a0: f90007e8    	str	x8, [sp, #0x8]
1000017a4: f85f03a1    	ldur	x1, [x29, #-0x10]
1000017a8: 910043e0    	add	x0, sp, #0x10
1000017ac: f90003e0    	str	x0, [sp]
1000017b0: 94000063    	bl	0x10000193c <__ZNSt3__18weak_ptrI4NodeEC1IS1_Li0EEERKNS_10shared_ptrIT_EE>
1000017b4: f94003e0    	ldr	x0, [sp]
1000017b8: f94007e1    	ldr	x1, [sp, #0x8]
1000017bc: 9400006d    	bl	0x100001970 <__ZNSt3__18weak_ptrI4NodeE4swapERS2_>
1000017c0: f94003e0    	ldr	x0, [sp]
1000017c4: 97ffff31    	bl	0x100001488 <__ZNSt3__18weak_ptrI4NodeED1Ev>
1000017c8: f94007e0    	ldr	x0, [sp, #0x8]
1000017cc: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000017d0: 910103ff    	add	sp, sp, #0x40
1000017d4: d65f03c0    	ret

00000001000017d8 <__ZNKSt3__119__shared_weak_count9use_countB8ne200100Ev>:
1000017d8: d10083ff    	sub	sp, sp, #0x20
1000017dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000017e0: 910043fd    	add	x29, sp, #0x10
1000017e4: f90007e0    	str	x0, [sp, #0x8]
1000017e8: f94007e0    	ldr	x0, [sp, #0x8]
1000017ec: 94000004    	bl	0x1000017fc <__ZNKSt3__114__shared_count9use_countB8ne200100Ev>
1000017f0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000017f4: 910083ff    	add	sp, sp, #0x20
1000017f8: d65f03c0    	ret

00000001000017fc <__ZNKSt3__114__shared_count9use_countB8ne200100Ev>:
1000017fc: d10083ff    	sub	sp, sp, #0x20
100001800: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001804: 910043fd    	add	x29, sp, #0x10
100001808: f90007e0    	str	x0, [sp, #0x8]
10000180c: f94007e8    	ldr	x8, [sp, #0x8]
100001810: 91002100    	add	x0, x8, #0x8
100001814: 94000009    	bl	0x100001838 <__ZNSt3__121__libcpp_relaxed_loadB8ne200100IlEET_PKS1_>
100001818: f90003e0    	str	x0, [sp]
10000181c: 14000001    	b	0x100001820 <__ZNKSt3__114__shared_count9use_countB8ne200100Ev+0x24>
100001820: f94003e8    	ldr	x8, [sp]
100001824: 91000500    	add	x0, x8, #0x1
100001828: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000182c: 910083ff    	add	sp, sp, #0x20
100001830: d65f03c0    	ret
100001834: 97fffee7    	bl	0x1000013d0 <___clang_call_terminate>

0000000100001838 <__ZNSt3__121__libcpp_relaxed_loadB8ne200100IlEET_PKS1_>:
100001838: d10043ff    	sub	sp, sp, #0x10
10000183c: f90007e0    	str	x0, [sp, #0x8]
100001840: f94007e8    	ldr	x8, [sp, #0x8]
100001844: f9400108    	ldr	x8, [x8]
100001848: f90003e8    	str	x8, [sp]
10000184c: f94003e0    	ldr	x0, [sp]
100001850: 910043ff    	add	sp, sp, #0x10
100001854: d65f03c0    	ret

0000000100001858 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_EERKNS0_IT_EEPS1_>:
100001858: d10103ff    	sub	sp, sp, #0x40
10000185c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100001860: 9100c3fd    	add	x29, sp, #0x30
100001864: f81f03a0    	stur	x0, [x29, #-0x10]
100001868: f9000fe1    	str	x1, [sp, #0x18]
10000186c: f9000be2    	str	x2, [sp, #0x10]
100001870: f85f03a8    	ldur	x8, [x29, #-0x10]
100001874: f90007e8    	str	x8, [sp, #0x8]
100001878: aa0803e9    	mov	x9, x8
10000187c: f81f83a9    	stur	x9, [x29, #-0x8]
100001880: f9400be9    	ldr	x9, [sp, #0x10]
100001884: f9000109    	str	x9, [x8]
100001888: f9400fe9    	ldr	x9, [sp, #0x18]
10000188c: f9400529    	ldr	x9, [x9, #0x8]
100001890: f9000509    	str	x9, [x8, #0x8]
100001894: f9400508    	ldr	x8, [x8, #0x8]
100001898: b40000c8    	cbz	x8, 0x1000018b0 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_EERKNS0_IT_EEPS1_+0x58>
10000189c: 14000001    	b	0x1000018a0 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_EERKNS0_IT_EEPS1_+0x48>
1000018a0: f94007e8    	ldr	x8, [sp, #0x8]
1000018a4: f9400500    	ldr	x0, [x8, #0x8]
1000018a8: 94000006    	bl	0x1000018c0 <__ZNSt3__119__shared_weak_count12__add_sharedB8ne200100Ev>
1000018ac: 14000001    	b	0x1000018b0 <__ZNSt3__110shared_ptrI4NodeEC2B8ne200100IS1_EERKNS0_IT_EEPS1_+0x58>
1000018b0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000018b4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000018b8: 910103ff    	add	sp, sp, #0x40
1000018bc: d65f03c0    	ret

00000001000018c0 <__ZNSt3__119__shared_weak_count12__add_sharedB8ne200100Ev>:
1000018c0: d10083ff    	sub	sp, sp, #0x20
1000018c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000018c8: 910043fd    	add	x29, sp, #0x10
1000018cc: f90007e0    	str	x0, [sp, #0x8]
1000018d0: f94007e0    	ldr	x0, [sp, #0x8]
1000018d4: 94000004    	bl	0x1000018e4 <__ZNSt3__114__shared_count12__add_sharedB8ne200100Ev>
1000018d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000018dc: 910083ff    	add	sp, sp, #0x20
1000018e0: d65f03c0    	ret

00000001000018e4 <__ZNSt3__114__shared_count12__add_sharedB8ne200100Ev>:
1000018e4: d10083ff    	sub	sp, sp, #0x20
1000018e8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000018ec: 910043fd    	add	x29, sp, #0x10
1000018f0: f90007e0    	str	x0, [sp, #0x8]
1000018f4: f94007e8    	ldr	x8, [sp, #0x8]
1000018f8: 91002100    	add	x0, x8, #0x8
1000018fc: 94000004    	bl	0x10000190c <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>
100001900: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001904: 910083ff    	add	sp, sp, #0x20
100001908: d65f03c0    	ret

000000010000190c <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>:
10000190c: d10083ff    	sub	sp, sp, #0x20
100001910: f9000fe0    	str	x0, [sp, #0x18]
100001914: f9400fe8    	ldr	x8, [sp, #0x18]
100001918: d2800029    	mov	x9, #0x1                ; =1
10000191c: f9000be9    	str	x9, [sp, #0x10]
100001920: f9400be9    	ldr	x9, [sp, #0x10]
100001924: f8290108    	ldadd	x9, x8, [x8]
100001928: 8b090108    	add	x8, x8, x9
10000192c: f90007e8    	str	x8, [sp, #0x8]
100001930: f94007e0    	ldr	x0, [sp, #0x8]
100001934: 910083ff    	add	sp, sp, #0x20
100001938: d65f03c0    	ret

000000010000193c <__ZNSt3__18weak_ptrI4NodeEC1IS1_Li0EEERKNS_10shared_ptrIT_EE>:
10000193c: d100c3ff    	sub	sp, sp, #0x30
100001940: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001944: 910083fd    	add	x29, sp, #0x20
100001948: f81f83a0    	stur	x0, [x29, #-0x8]
10000194c: f9000be1    	str	x1, [sp, #0x10]
100001950: f85f83a0    	ldur	x0, [x29, #-0x8]
100001954: f90007e0    	str	x0, [sp, #0x8]
100001958: f9400be1    	ldr	x1, [sp, #0x10]
10000195c: 94000016    	bl	0x1000019b4 <__ZNSt3__18weak_ptrI4NodeEC2IS1_Li0EEERKNS_10shared_ptrIT_EE>
100001960: f94007e0    	ldr	x0, [sp, #0x8]
100001964: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001968: 9100c3ff    	add	sp, sp, #0x30
10000196c: d65f03c0    	ret

0000000100001970 <__ZNSt3__18weak_ptrI4NodeE4swapERS2_>:
100001970: d100c3ff    	sub	sp, sp, #0x30
100001974: a9027bfd    	stp	x29, x30, [sp, #0x20]
100001978: 910083fd    	add	x29, sp, #0x20
10000197c: f81f83a0    	stur	x0, [x29, #-0x8]
100001980: f9000be1    	str	x1, [sp, #0x10]
100001984: f85f83a0    	ldur	x0, [x29, #-0x8]
100001988: f90007e0    	str	x0, [sp, #0x8]
10000198c: f9400be1    	ldr	x1, [sp, #0x10]
100001990: 9400002d    	bl	0x100001a44 <__ZNSt3__14swapB8ne200100IP4NodeEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS4_EE5valueEvE4typeERS4_S7_>
100001994: f94007e9    	ldr	x9, [sp, #0x8]
100001998: f9400be8    	ldr	x8, [sp, #0x10]
10000199c: 91002120    	add	x0, x9, #0x8
1000019a0: 91002101    	add	x1, x8, #0x8
1000019a4: 94000037    	bl	0x100001a80 <__ZNSt3__14swapB8ne200100IPNS_19__shared_weak_countEEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS4_EE5valueEvE4typeERS4_S7_>
1000019a8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000019ac: 9100c3ff    	add	sp, sp, #0x30
1000019b0: d65f03c0    	ret

00000001000019b4 <__ZNSt3__18weak_ptrI4NodeEC2IS1_Li0EEERKNS_10shared_ptrIT_EE>:
1000019b4: d100c3ff    	sub	sp, sp, #0x30
1000019b8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000019bc: 910083fd    	add	x29, sp, #0x20
1000019c0: f9000be0    	str	x0, [sp, #0x10]
1000019c4: f90007e1    	str	x1, [sp, #0x8]
1000019c8: f9400be8    	ldr	x8, [sp, #0x10]
1000019cc: f90003e8    	str	x8, [sp]
1000019d0: aa0803e9    	mov	x9, x8
1000019d4: f81f83a9    	stur	x9, [x29, #-0x8]
1000019d8: f94007e9    	ldr	x9, [sp, #0x8]
1000019dc: f9400129    	ldr	x9, [x9]
1000019e0: f9000109    	str	x9, [x8]
1000019e4: f94007e9    	ldr	x9, [sp, #0x8]
1000019e8: f9400529    	ldr	x9, [x9, #0x8]
1000019ec: f9000509    	str	x9, [x8, #0x8]
1000019f0: f9400508    	ldr	x8, [x8, #0x8]
1000019f4: b40000c8    	cbz	x8, 0x100001a0c <__ZNSt3__18weak_ptrI4NodeEC2IS1_Li0EEERKNS_10shared_ptrIT_EE+0x58>
1000019f8: 14000001    	b	0x1000019fc <__ZNSt3__18weak_ptrI4NodeEC2IS1_Li0EEERKNS_10shared_ptrIT_EE+0x48>
1000019fc: f94003e8    	ldr	x8, [sp]
100001a00: f9400500    	ldr	x0, [x8, #0x8]
100001a04: 94000006    	bl	0x100001a1c <__ZNSt3__119__shared_weak_count10__add_weakB8ne200100Ev>
100001a08: 14000001    	b	0x100001a0c <__ZNSt3__18weak_ptrI4NodeEC2IS1_Li0EEERKNS_10shared_ptrIT_EE+0x58>
100001a0c: f85f83a0    	ldur	x0, [x29, #-0x8]
100001a10: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100001a14: 9100c3ff    	add	sp, sp, #0x30
100001a18: d65f03c0    	ret

0000000100001a1c <__ZNSt3__119__shared_weak_count10__add_weakB8ne200100Ev>:
100001a1c: d10083ff    	sub	sp, sp, #0x20
100001a20: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001a24: 910043fd    	add	x29, sp, #0x10
100001a28: f90007e0    	str	x0, [sp, #0x8]
100001a2c: f94007e8    	ldr	x8, [sp, #0x8]
100001a30: 91004100    	add	x0, x8, #0x10
100001a34: 97ffffb6    	bl	0x10000190c <__ZNSt3__134__libcpp_atomic_refcount_incrementB8ne200100IlEET_RS1_>
100001a38: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001a3c: 910083ff    	add	sp, sp, #0x20
100001a40: d65f03c0    	ret

0000000100001a44 <__ZNSt3__14swapB8ne200100IP4NodeEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS4_EE5valueEvE4typeERS4_S7_>:
100001a44: d10083ff    	sub	sp, sp, #0x20
100001a48: f9000fe0    	str	x0, [sp, #0x18]
100001a4c: f9000be1    	str	x1, [sp, #0x10]
100001a50: f9400fe8    	ldr	x8, [sp, #0x18]
100001a54: f9400108    	ldr	x8, [x8]
100001a58: f90007e8    	str	x8, [sp, #0x8]
100001a5c: f9400be8    	ldr	x8, [sp, #0x10]
100001a60: f9400108    	ldr	x8, [x8]
100001a64: f9400fe9    	ldr	x9, [sp, #0x18]
100001a68: f9000128    	str	x8, [x9]
100001a6c: f94007e8    	ldr	x8, [sp, #0x8]
100001a70: f9400be9    	ldr	x9, [sp, #0x10]
100001a74: f9000128    	str	x8, [x9]
100001a78: 910083ff    	add	sp, sp, #0x20
100001a7c: d65f03c0    	ret

0000000100001a80 <__ZNSt3__14swapB8ne200100IPNS_19__shared_weak_countEEENS_9enable_ifIXaasr21is_move_constructibleIT_EE5valuesr18is_move_assignableIS4_EE5valueEvE4typeERS4_S7_>:
100001a80: d10083ff    	sub	sp, sp, #0x20
100001a84: f9000fe0    	str	x0, [sp, #0x18]
100001a88: f9000be1    	str	x1, [sp, #0x10]
100001a8c: f9400fe8    	ldr	x8, [sp, #0x18]
100001a90: f9400108    	ldr	x8, [x8]
100001a94: f90007e8    	str	x8, [sp, #0x8]
100001a98: f9400be8    	ldr	x8, [sp, #0x10]
100001a9c: f9400108    	ldr	x8, [x8]
100001aa0: f9400fe9    	ldr	x9, [sp, #0x18]
100001aa4: f9000128    	str	x8, [x9]
100001aa8: f94007e8    	ldr	x8, [sp, #0x8]
100001aac: f9400be9    	ldr	x9, [sp, #0x10]
100001ab0: f9000128    	str	x8, [x9]
100001ab4: 910083ff    	add	sp, sp, #0x20
100001ab8: d65f03c0    	ret

0000000100001abc <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE8_Storage10__get_elemB8ne200100Ev>:
100001abc: d10043ff    	sub	sp, sp, #0x10
100001ac0: f90007e0    	str	x0, [sp, #0x8]
100001ac4: f94007e0    	ldr	x0, [sp, #0x8]
100001ac8: 910043ff    	add	sp, sp, #0x10
100001acc: d65f03c0    	ret

0000000100001ad0 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEED2B8ne200100Ev>:
100001ad0: d10083ff    	sub	sp, sp, #0x20
100001ad4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001ad8: 910043fd    	add	x29, sp, #0x10
100001adc: f90007e0    	str	x0, [sp, #0x8]
100001ae0: f94007e0    	ldr	x0, [sp, #0x8]
100001ae4: f90003e0    	str	x0, [sp]
100001ae8: 94000005    	bl	0x100001afc <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE9__destroyB8ne200100Ev>
100001aec: f94003e0    	ldr	x0, [sp]
100001af0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001af4: 910083ff    	add	sp, sp, #0x20
100001af8: d65f03c0    	ret

0000000100001afc <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE9__destroyB8ne200100Ev>:
100001afc: d10083ff    	sub	sp, sp, #0x20
100001b00: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001b04: 910043fd    	add	x29, sp, #0x10
100001b08: f90007e0    	str	x0, [sp, #0x8]
100001b0c: f94007e8    	ldr	x8, [sp, #0x8]
100001b10: f90003e8    	str	x8, [sp]
100001b14: f9400908    	ldr	x8, [x8, #0x10]
100001b18: b40000e8    	cbz	x8, 0x100001b34 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE9__destroyB8ne200100Ev+0x38>
100001b1c: 14000001    	b	0x100001b20 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE9__destroyB8ne200100Ev+0x24>
100001b20: f94003e0    	ldr	x0, [sp]
100001b24: f9400801    	ldr	x1, [x0, #0x10]
100001b28: f9400402    	ldr	x2, [x0, #0x8]
100001b2c: 97fffe75    	bl	0x100001500 <__ZNSt3__116allocator_traitsINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE10deallocateB8ne200100ERS6_PS5_m>
100001b30: 14000001    	b	0x100001b34 <__ZNSt3__118__allocation_guardINS_9allocatorINS_20__shared_ptr_emplaceI4NodeNS1_IS3_EEEEEEE9__destroyB8ne200100Ev+0x38>
100001b34: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001b38: 910083ff    	add	sp, sp, #0x20
100001b3c: d65f03c0    	ret

0000000100001b40 <__ZNSt3__19allocatorI4NodeEC2B8ne200100Ev>:
100001b40: d10083ff    	sub	sp, sp, #0x20
100001b44: a9017bfd    	stp	x29, x30, [sp, #0x10]
100001b48: 910043fd    	add	x29, sp, #0x10
100001b4c: f90007e0    	str	x0, [sp, #0x8]
100001b50: f94007e0    	ldr	x0, [sp, #0x8]
100001b54: f90003e0    	str	x0, [sp]
100001b58: 94000005    	bl	0x100001b6c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorI4NodeEEEC2B8ne200100Ev>
100001b5c: f94003e0    	ldr	x0, [sp]
100001b60: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100001b64: 910083ff    	add	sp, sp, #0x20
100001b68: d65f03c0    	ret

0000000100001b6c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorI4NodeEEEC2B8ne200100Ev>:
100001b6c: d10043ff    	sub	sp, sp, #0x10
100001b70: f90007e0    	str	x0, [sp, #0x8]
100001b74: f94007e0    	ldr	x0, [sp, #0x8]
100001b78: 910043ff    	add	sp, sp, #0x10
100001b7c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100001b80 <__stubs>:
100001b80: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001b84: f9400610    	ldr	x16, [x16, #0x8]
100001b88: d61f0200    	br	x16
100001b8c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001b90: f9400a10    	ldr	x16, [x16, #0x10]
100001b94: d61f0200    	br	x16
100001b98: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001b9c: f9400e10    	ldr	x16, [x16, #0x18]
100001ba0: d61f0200    	br	x16
100001ba4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001ba8: f9401210    	ldr	x16, [x16, #0x20]
100001bac: d61f0200    	br	x16
100001bb0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001bb4: f9401e10    	ldr	x16, [x16, #0x38]
100001bb8: d61f0200    	br	x16
100001bbc: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001bc0: f9402a10    	ldr	x16, [x16, #0x50]
100001bc4: d61f0200    	br	x16
100001bc8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001bcc: f9403210    	ldr	x16, [x16, #0x60]
100001bd0: d61f0200    	br	x16
100001bd4: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001bd8: f9403610    	ldr	x16, [x16, #0x68]
100001bdc: d61f0200    	br	x16
100001be0: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001be4: f9403a10    	ldr	x16, [x16, #0x70]
100001be8: d61f0200    	br	x16
100001bec: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001bf0: f9403e10    	ldr	x16, [x16, #0x78]
100001bf4: d61f0200    	br	x16
100001bf8: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001bfc: f9404210    	ldr	x16, [x16, #0x80]
100001c00: d61f0200    	br	x16
100001c04: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c08: f9404e10    	ldr	x16, [x16, #0x98]
100001c0c: d61f0200    	br	x16
100001c10: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c14: f9405210    	ldr	x16, [x16, #0xa0]
100001c18: d61f0200    	br	x16
100001c1c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c20: f9405610    	ldr	x16, [x16, #0xa8]
100001c24: d61f0200    	br	x16
100001c28: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c2c: f9405a10    	ldr	x16, [x16, #0xb0]
100001c30: d61f0200    	br	x16
100001c34: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c38: f9406210    	ldr	x16, [x16, #0xc0]
100001c3c: d61f0200    	br	x16
100001c40: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c44: f9406610    	ldr	x16, [x16, #0xc8]
100001c48: d61f0200    	br	x16
100001c4c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c50: f9406e10    	ldr	x16, [x16, #0xd8]
100001c54: d61f0200    	br	x16
100001c58: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c5c: f9407210    	ldr	x16, [x16, #0xe0]
100001c60: d61f0200    	br	x16
100001c64: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c68: f9407610    	ldr	x16, [x16, #0xe8]
100001c6c: d61f0200    	br	x16
100001c70: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c74: f9407a10    	ldr	x16, [x16, #0xf0]
100001c78: d61f0200    	br	x16
100001c7c: f0000010    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100001c80: f9407e10    	ldr	x16, [x16, #0xf8]
100001c84: d61f0200    	br	x16
