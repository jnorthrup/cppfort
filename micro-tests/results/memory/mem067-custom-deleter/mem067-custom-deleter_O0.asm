
/Users/jim/work/cppfort/micro-tests/results/memory/mem067-custom-deleter/mem067-custom-deleter_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z14custom_deleterPi>:
100000538: d10083ff    	sub	sp, sp, #0x20
10000053c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000540: 910043fd    	add	x29, sp, #0x10
100000544: f90007e0    	str	x0, [sp, #0x8]
100000548: f94007e8    	ldr	x8, [sp, #0x8]
10000054c: f90003e8    	str	x8, [sp]
100000550: b40000a8    	cbz	x8, 0x100000564 <__Z14custom_deleterPi+0x2c>
100000554: 14000001    	b	0x100000558 <__Z14custom_deleterPi+0x20>
100000558: f94003e0    	ldr	x0, [sp]
10000055c: 94000236    	bl	0x100000e34 <_strcmp+0x100000e34>
100000560: 14000001    	b	0x100000564 <__Z14custom_deleterPi+0x2c>
100000564: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000568: 910083ff    	add	sp, sp, #0x20
10000056c: d65f03c0    	ret

0000000100000570 <__Z19test_custom_deleterv>:
100000570: d100c3ff    	sub	sp, sp, #0x30
100000574: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000578: 910083fd    	add	x29, sp, #0x20
10000057c: d2800080    	mov	x0, #0x4                ; =4
100000580: 94000230    	bl	0x100000e40 <_strcmp+0x100000e40>
100000584: aa0003e1    	mov	x1, x0
100000588: 52800548    	mov	w8, #0x2a               ; =42
10000058c: b9000028    	str	w8, [x1]
100000590: 910043e0    	add	x0, sp, #0x10
100000594: f90003e0    	str	x0, [sp]
100000598: 90000002    	adrp	x2, 0x100000000 <_strcmp+0x100000000>
10000059c: 9114e042    	add	x2, x2, #0x538
1000005a0: 9400000c    	bl	0x1000005d0 <__ZNSt3__110shared_ptrIiEC1B8ne200100IiPFvPiELi0EEEPT_T0_>
1000005a4: f94003e0    	ldr	x0, [sp]
1000005a8: 94000019    	bl	0x10000060c <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>
1000005ac: aa0003e8    	mov	x8, x0
1000005b0: f94003e0    	ldr	x0, [sp]
1000005b4: b9400108    	ldr	w8, [x8]
1000005b8: b9000fe8    	str	w8, [sp, #0xc]
1000005bc: 9400001a    	bl	0x100000624 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>
1000005c0: b9400fe0    	ldr	w0, [sp, #0xc]
1000005c4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000005c8: 9100c3ff    	add	sp, sp, #0x30
1000005cc: d65f03c0    	ret

00000001000005d0 <__ZNSt3__110shared_ptrIiEC1B8ne200100IiPFvPiELi0EEEPT_T0_>:
1000005d0: d100c3ff    	sub	sp, sp, #0x30
1000005d4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005d8: 910083fd    	add	x29, sp, #0x20
1000005dc: f81f83a0    	stur	x0, [x29, #-0x8]
1000005e0: f9000be1    	str	x1, [sp, #0x10]
1000005e4: f90007e2    	str	x2, [sp, #0x8]
1000005e8: f85f83a0    	ldur	x0, [x29, #-0x8]
1000005ec: f90003e0    	str	x0, [sp]
1000005f0: f9400be1    	ldr	x1, [sp, #0x10]
1000005f4: f94007e2    	ldr	x2, [sp, #0x8]
1000005f8: 9400001e    	bl	0x100000670 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_>
1000005fc: f94003e0    	ldr	x0, [sp]
100000600: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000604: 9100c3ff    	add	sp, sp, #0x30
100000608: d65f03c0    	ret

000000010000060c <__ZNKSt3__110shared_ptrIiEdeB8ne200100Ev>:
10000060c: d10043ff    	sub	sp, sp, #0x10
100000610: f90007e0    	str	x0, [sp, #0x8]
100000614: f94007e8    	ldr	x8, [sp, #0x8]
100000618: f9400100    	ldr	x0, [x8]
10000061c: 910043ff    	add	sp, sp, #0x10
100000620: d65f03c0    	ret

0000000100000624 <__ZNSt3__110shared_ptrIiED1B8ne200100Ev>:
100000624: d10083ff    	sub	sp, sp, #0x20
100000628: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000062c: 910043fd    	add	x29, sp, #0x10
100000630: f90007e0    	str	x0, [sp, #0x8]
100000634: f94007e0    	ldr	x0, [sp, #0x8]
100000638: f90003e0    	str	x0, [sp]
10000063c: 940001b2    	bl	0x100000d04 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>
100000640: f94003e0    	ldr	x0, [sp]
100000644: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000648: 910083ff    	add	sp, sp, #0x20
10000064c: d65f03c0    	ret

0000000100000650 <_main>:
100000650: d10083ff    	sub	sp, sp, #0x20
100000654: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000658: 910043fd    	add	x29, sp, #0x10
10000065c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000660: 97ffffc4    	bl	0x100000570 <__Z19test_custom_deleterv>
100000664: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000668: 910083ff    	add	sp, sp, #0x20
10000066c: d65f03c0    	ret

0000000100000670 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_>:
100000670: d101c3ff    	sub	sp, sp, #0x70
100000674: a9067bfd    	stp	x29, x30, [sp, #0x60]
100000678: 910183fd    	add	x29, sp, #0x60
10000067c: f81f83a0    	stur	x0, [x29, #-0x8]
100000680: f81f03a1    	stur	x1, [x29, #-0x10]
100000684: f81e83a2    	stur	x2, [x29, #-0x18]
100000688: f85f83a9    	ldur	x9, [x29, #-0x8]
10000068c: f90017e9    	str	x9, [sp, #0x28]
100000690: f85f03a8    	ldur	x8, [x29, #-0x10]
100000694: f9000128    	str	x8, [x9]
100000698: d2800500    	mov	x0, #0x28               ; =40
10000069c: 940001e9    	bl	0x100000e40 <_strcmp+0x100000e40>
1000006a0: f9001be0    	str	x0, [sp, #0x30]
1000006a4: 14000001    	b	0x1000006a8 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0x38>
1000006a8: f85f03a8    	ldur	x8, [x29, #-0x10]
1000006ac: f9000fe8    	str	x8, [sp, #0x18]
1000006b0: f85e83a8    	ldur	x8, [x29, #-0x18]
1000006b4: f90013e8    	str	x8, [sp, #0x20]
1000006b8: d10097a0    	sub	x0, x29, #0x25
1000006bc: 94000031    	bl	0x100000780 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>
1000006c0: f9401be0    	ldr	x0, [sp, #0x30]
1000006c4: f9400fe1    	ldr	x1, [sp, #0x18]
1000006c8: f94013e2    	ldr	x2, [sp, #0x20]
1000006cc: 94000038    	bl	0x1000007ac <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEEC1B8ne200100ES1_S3_S5_>
1000006d0: 14000001    	b	0x1000006d4 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0x64>
1000006d4: f94017e0    	ldr	x0, [sp, #0x28]
1000006d8: f9401be8    	ldr	x8, [sp, #0x30]
1000006dc: f9000408    	str	x8, [x0, #0x8]
1000006e0: f85f03aa    	ldur	x10, [x29, #-0x10]
1000006e4: f85f03a8    	ldur	x8, [x29, #-0x10]
1000006e8: 910003e9    	mov	x9, sp
1000006ec: f900012a    	str	x10, [x9]
1000006f0: f9000528    	str	x8, [x9, #0x8]
1000006f4: 9400003d    	bl	0x1000007e8 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>
1000006f8: 1400001a    	b	0x100000760 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0xf0>
1000006fc: f81e03a0    	stur	x0, [x29, #-0x20]
100000700: aa0103e8    	mov	x8, x1
100000704: b81dc3a8    	stur	w8, [x29, #-0x24]
100000708: 14000008    	b	0x100000728 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0xb8>
10000070c: aa0003e8    	mov	x8, x0
100000710: f9401be0    	ldr	x0, [sp, #0x30]
100000714: f81e03a8    	stur	x8, [x29, #-0x20]
100000718: aa0103e8    	mov	x8, x1
10000071c: b81dc3a8    	stur	w8, [x29, #-0x24]
100000720: 940001c5    	bl	0x100000e34 <_strcmp+0x100000e34>
100000724: 14000001    	b	0x100000728 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0xb8>
100000728: f85e03a0    	ldur	x0, [x29, #-0x20]
10000072c: 940001c8    	bl	0x100000e4c <_strcmp+0x100000e4c>
100000730: f85e83a8    	ldur	x8, [x29, #-0x18]
100000734: f85f03a0    	ldur	x0, [x29, #-0x10]
100000738: d63f0100    	blr	x8
10000073c: 14000001    	b	0x100000740 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0xd0>
100000740: 940001c6    	bl	0x100000e58 <_strcmp+0x100000e58>
100000744: 1400000e    	b	0x10000077c <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0x10c>
100000748: f81e03a0    	stur	x0, [x29, #-0x20]
10000074c: aa0103e8    	mov	x8, x1
100000750: b81dc3a8    	stur	w8, [x29, #-0x24]
100000754: 940001c4    	bl	0x100000e64 <_strcmp+0x100000e64>
100000758: 14000001    	b	0x10000075c <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0xec>
10000075c: 14000005    	b	0x100000770 <__ZNSt3__110shared_ptrIiEC2B8ne200100IiPFvPiELi0EEEPT_T0_+0x100>
100000760: f94017e0    	ldr	x0, [sp, #0x28]
100000764: a9467bfd    	ldp	x29, x30, [sp, #0x60]
100000768: 9101c3ff    	add	sp, sp, #0x70
10000076c: d65f03c0    	ret
100000770: f85e03a0    	ldur	x0, [x29, #-0x20]
100000774: 940001bf    	bl	0x100000e70 <_strcmp+0x100000e70>
100000778: 94000020    	bl	0x1000007f8 <___clang_call_terminate>
10000077c: d4200020    	brk	#0x1

0000000100000780 <__ZNSt3__19allocatorIiEC1B8ne200100Ev>:
100000780: d10083ff    	sub	sp, sp, #0x20
100000784: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000788: 910043fd    	add	x29, sp, #0x10
10000078c: f90007e0    	str	x0, [sp, #0x8]
100000790: f94007e0    	ldr	x0, [sp, #0x8]
100000794: f90003e0    	str	x0, [sp]
100000798: 9400001c    	bl	0x100000808 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>
10000079c: f94003e0    	ldr	x0, [sp]
1000007a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000007a4: 910083ff    	add	sp, sp, #0x20
1000007a8: d65f03c0    	ret

00000001000007ac <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEEC1B8ne200100ES1_S3_S5_>:
1000007ac: d10103ff    	sub	sp, sp, #0x40
1000007b0: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000007b4: 9100c3fd    	add	x29, sp, #0x30
1000007b8: f81f03a0    	stur	x0, [x29, #-0x10]
1000007bc: f9000fe1    	str	x1, [sp, #0x18]
1000007c0: f9000be2    	str	x2, [sp, #0x10]
1000007c4: f85f03a0    	ldur	x0, [x29, #-0x10]
1000007c8: f90007e0    	str	x0, [sp, #0x8]
1000007cc: f9400fe1    	ldr	x1, [sp, #0x18]
1000007d0: f9400be2    	ldr	x2, [sp, #0x10]
1000007d4: 9400001d    	bl	0x100000848 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEEC2B8ne200100ES1_S3_S5_>
1000007d8: f94007e0    	ldr	x0, [sp, #0x8]
1000007dc: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000007e0: 910103ff    	add	sp, sp, #0x40
1000007e4: d65f03c0    	ret

00000001000007e8 <__ZNSt3__110shared_ptrIiE18__enable_weak_thisB8ne200100Ez>:
1000007e8: d10043ff    	sub	sp, sp, #0x10
1000007ec: f90007e0    	str	x0, [sp, #0x8]
1000007f0: 910043ff    	add	sp, sp, #0x10
1000007f4: d65f03c0    	ret

00000001000007f8 <___clang_call_terminate>:
1000007f8: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000007fc: 910003fd    	mov	x29, sp
100000800: 94000193    	bl	0x100000e4c <_strcmp+0x100000e4c>
100000804: 9400019e    	bl	0x100000e7c <_strcmp+0x100000e7c>

0000000100000808 <__ZNSt3__19allocatorIiEC2B8ne200100Ev>:
100000808: d10083ff    	sub	sp, sp, #0x20
10000080c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000810: 910043fd    	add	x29, sp, #0x10
100000814: f90007e0    	str	x0, [sp, #0x8]
100000818: f94007e0    	ldr	x0, [sp, #0x8]
10000081c: f90003e0    	str	x0, [sp]
100000820: 94000005    	bl	0x100000834 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>
100000824: f94003e0    	ldr	x0, [sp]
100000828: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000082c: 910083ff    	add	sp, sp, #0x20
100000830: d65f03c0    	ret

0000000100000834 <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIiEEEC2B8ne200100Ev>:
100000834: d10043ff    	sub	sp, sp, #0x10
100000838: f90007e0    	str	x0, [sp, #0x8]
10000083c: f94007e0    	ldr	x0, [sp, #0x8]
100000840: 910043ff    	add	sp, sp, #0x10
100000844: d65f03c0    	ret

0000000100000848 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEEC2B8ne200100ES1_S3_S5_>:
100000848: d10103ff    	sub	sp, sp, #0x40
10000084c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000850: 9100c3fd    	add	x29, sp, #0x30
100000854: f81f03a0    	stur	x0, [x29, #-0x10]
100000858: f9000fe1    	str	x1, [sp, #0x18]
10000085c: f9000be2    	str	x2, [sp, #0x10]
100000860: f85f03a0    	ldur	x0, [x29, #-0x10]
100000864: f90007e0    	str	x0, [sp, #0x8]
100000868: d2800001    	mov	x1, #0x0                ; =0
10000086c: 9400000d    	bl	0x1000008a0 <__ZNSt3__119__shared_weak_countC2B8ne200100El>
100000870: f94007e0    	ldr	x0, [sp, #0x8]
100000874: 90000028    	adrp	x8, 0x100004000 <_strcmp+0x100004000>
100000878: 9101c108    	add	x8, x8, #0x70
10000087c: 91004108    	add	x8, x8, #0x10
100000880: f9000008    	str	x8, [x0]
100000884: f9400fe8    	ldr	x8, [sp, #0x18]
100000888: f9000c08    	str	x8, [x0, #0x18]
10000088c: f9400be8    	ldr	x8, [sp, #0x10]
100000890: f9001008    	str	x8, [x0, #0x20]
100000894: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000898: 910103ff    	add	sp, sp, #0x40
10000089c: d65f03c0    	ret

00000001000008a0 <__ZNSt3__119__shared_weak_countC2B8ne200100El>:
1000008a0: d100c3ff    	sub	sp, sp, #0x30
1000008a4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000008a8: 910083fd    	add	x29, sp, #0x20
1000008ac: f81f83a0    	stur	x0, [x29, #-0x8]
1000008b0: f9000be1    	str	x1, [sp, #0x10]
1000008b4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000008b8: f90007e0    	str	x0, [sp, #0x8]
1000008bc: f9400be1    	ldr	x1, [sp, #0x10]
1000008c0: 94000059    	bl	0x100000a24 <__ZNSt3__114__shared_countC2B8ne200100El>
1000008c4: f94007e0    	ldr	x0, [sp, #0x8]
1000008c8: 90000028    	adrp	x8, 0x100004000 <_strcmp+0x100004000>
1000008cc: f9402108    	ldr	x8, [x8, #0x40]
1000008d0: 91004108    	add	x8, x8, #0x10
1000008d4: f9000008    	str	x8, [x0]
1000008d8: f9400be8    	ldr	x8, [sp, #0x10]
1000008dc: f9000808    	str	x8, [x0, #0x10]
1000008e0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000008e4: 9100c3ff    	add	sp, sp, #0x30
1000008e8: d65f03c0    	ret

00000001000008ec <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED1Ev>:
1000008ec: d10083ff    	sub	sp, sp, #0x20
1000008f0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000008f4: 910043fd    	add	x29, sp, #0x10
1000008f8: f90007e0    	str	x0, [sp, #0x8]
1000008fc: f94007e0    	ldr	x0, [sp, #0x8]
100000900: f90003e0    	str	x0, [sp]
100000904: 94000054    	bl	0x100000a54 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED2Ev>
100000908: f94003e0    	ldr	x0, [sp]
10000090c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000910: 910083ff    	add	sp, sp, #0x20
100000914: d65f03c0    	ret

0000000100000918 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED0Ev>:
100000918: d10083ff    	sub	sp, sp, #0x20
10000091c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000920: 910043fd    	add	x29, sp, #0x10
100000924: f90007e0    	str	x0, [sp, #0x8]
100000928: f94007e0    	ldr	x0, [sp, #0x8]
10000092c: f90003e0    	str	x0, [sp]
100000930: 97ffffef    	bl	0x1000008ec <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED1Ev>
100000934: f94003e0    	ldr	x0, [sp]
100000938: 9400013f    	bl	0x100000e34 <_strcmp+0x100000e34>
10000093c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000940: 910083ff    	add	sp, sp, #0x20
100000944: d65f03c0    	ret

0000000100000948 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE16__on_zero_sharedEv>:
100000948: d10083ff    	sub	sp, sp, #0x20
10000094c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000950: 910043fd    	add	x29, sp, #0x10
100000954: f90007e0    	str	x0, [sp, #0x8]
100000958: f94007e9    	ldr	x9, [sp, #0x8]
10000095c: f9401128    	ldr	x8, [x9, #0x20]
100000960: f9400d20    	ldr	x0, [x9, #0x18]
100000964: d63f0100    	blr	x8
100000968: 14000001    	b	0x10000096c <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE16__on_zero_sharedEv+0x24>
10000096c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000970: 910083ff    	add	sp, sp, #0x20
100000974: d65f03c0    	ret
100000978: 97ffffa0    	bl	0x1000007f8 <___clang_call_terminate>

000000010000097c <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info>:
10000097c: d100c3ff    	sub	sp, sp, #0x30
100000980: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000984: 910083fd    	add	x29, sp, #0x20
100000988: f81f83a0    	stur	x0, [x29, #-0x8]
10000098c: f9000be1    	str	x1, [sp, #0x10]
100000990: f85f83a8    	ldur	x8, [x29, #-0x8]
100000994: f90007e8    	str	x8, [sp, #0x8]
100000998: f9400be0    	ldr	x0, [sp, #0x10]
10000099c: 90000021    	adrp	x1, 0x100004000 <_strcmp+0x100004000>
1000009a0: 91034021    	add	x1, x1, #0xd0
1000009a4: 94000037    	bl	0x100000a80 <__ZNKSt9type_infoeqB8ne200100ERKS_>
1000009a8: 360000c0    	tbz	w0, #0x0, 0x1000009c0 <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x44>
1000009ac: 14000001    	b	0x1000009b0 <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x34>
1000009b0: f94007e8    	ldr	x8, [sp, #0x8]
1000009b4: 91008108    	add	x8, x8, #0x20
1000009b8: f90003e8    	str	x8, [sp]
1000009bc: 14000004    	b	0x1000009cc <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x50>
1000009c0: d2800008    	mov	x8, #0x0                ; =0
1000009c4: f90003e8    	str	x8, [sp]
1000009c8: 14000001    	b	0x1000009cc <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x50>
1000009cc: f94003e0    	ldr	x0, [sp]
1000009d0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000009d4: 9100c3ff    	add	sp, sp, #0x30
1000009d8: d65f03c0    	ret

00000001000009dc <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE21__on_zero_shared_weakEv>:
1000009dc: d100c3ff    	sub	sp, sp, #0x30
1000009e0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000009e4: 910083fd    	add	x29, sp, #0x20
1000009e8: f81f83a0    	stur	x0, [x29, #-0x8]
1000009ec: f85f83a1    	ldur	x1, [x29, #-0x8]
1000009f0: f90003e1    	str	x1, [sp]
1000009f4: d10027a0    	sub	x0, x29, #0x9
1000009f8: f90007e0    	str	x0, [sp, #0x8]
1000009fc: 9400005d    	bl	0x100000b70 <__ZNSt3__19allocatorINS_20__shared_ptr_pointerIPiPFvS2_ENS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>
100000a00: f94003e0    	ldr	x0, [sp]
100000a04: 94000075    	bl	0x100000bd8 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEE10pointer_toB8ne200100ERS7_>
100000a08: aa0003e1    	mov	x1, x0
100000a0c: f94007e0    	ldr	x0, [sp, #0x8]
100000a10: d2800022    	mov	x2, #0x1                ; =1
100000a14: 94000064    	bl	0x100000ba4 <__ZNSt3__19allocatorINS_20__shared_ptr_pointerIPiPFvS2_ENS0_IiEEEEE10deallocateB8ne200100EPS6_m>
100000a18: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000a1c: 9100c3ff    	add	sp, sp, #0x30
100000a20: d65f03c0    	ret

0000000100000a24 <__ZNSt3__114__shared_countC2B8ne200100El>:
100000a24: d10043ff    	sub	sp, sp, #0x10
100000a28: f90007e0    	str	x0, [sp, #0x8]
100000a2c: f90003e1    	str	x1, [sp]
100000a30: f94007e0    	ldr	x0, [sp, #0x8]
100000a34: 90000028    	adrp	x8, 0x100004000 <_strcmp+0x100004000>
100000a38: f9402508    	ldr	x8, [x8, #0x48]
100000a3c: 91004108    	add	x8, x8, #0x10
100000a40: f9000008    	str	x8, [x0]
100000a44: f94003e8    	ldr	x8, [sp]
100000a48: f9000408    	str	x8, [x0, #0x8]
100000a4c: 910043ff    	add	sp, sp, #0x10
100000a50: d65f03c0    	ret

0000000100000a54 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED2Ev>:
100000a54: d10083ff    	sub	sp, sp, #0x20
100000a58: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000a5c: 910043fd    	add	x29, sp, #0x10
100000a60: f90007e0    	str	x0, [sp, #0x8]
100000a64: f94007e0    	ldr	x0, [sp, #0x8]
100000a68: f90003e0    	str	x0, [sp]
100000a6c: 94000107    	bl	0x100000e88 <_strcmp+0x100000e88>
100000a70: f94003e0    	ldr	x0, [sp]
100000a74: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000a78: 910083ff    	add	sp, sp, #0x20
100000a7c: d65f03c0    	ret

0000000100000a80 <__ZNKSt9type_infoeqB8ne200100ERKS_>:
100000a80: d10143ff    	sub	sp, sp, #0x50
100000a84: a9047bfd    	stp	x29, x30, [sp, #0x40]
100000a88: 910103fd    	add	x29, sp, #0x40
100000a8c: f9000be0    	str	x0, [sp, #0x10]
100000a90: f90007e1    	str	x1, [sp, #0x8]
100000a94: f9400be8    	ldr	x8, [sp, #0x10]
100000a98: f9400509    	ldr	x9, [x8, #0x8]
100000a9c: f94007e8    	ldr	x8, [sp, #0x8]
100000aa0: f9400508    	ldr	x8, [x8, #0x8]
100000aa4: f90013e9    	str	x9, [sp, #0x20]
100000aa8: f9000fe8    	str	x8, [sp, #0x18]
100000aac: f94013e8    	ldr	x8, [sp, #0x20]
100000ab0: f9400fe9    	ldr	x9, [sp, #0x18]
100000ab4: eb090108    	subs	x8, x8, x9
100000ab8: 540000e1    	b.ne	0x100000ad4 <__ZNKSt9type_infoeqB8ne200100ERKS_+0x54>
100000abc: 14000001    	b	0x100000ac0 <__ZNKSt9type_infoeqB8ne200100ERKS_+0x40>
100000ac0: 52800028    	mov	w8, #0x1                ; =1
100000ac4: 12000108    	and	w8, w8, #0x1
100000ac8: 12000108    	and	w8, w8, #0x1
100000acc: 381ef3a8    	sturb	w8, [x29, #-0x11]
100000ad0: 1400001c    	b	0x100000b40 <__ZNKSt9type_infoeqB8ne200100ERKS_+0xc0>
100000ad4: f94013e0    	ldr	x0, [sp, #0x20]
100000ad8: 9400001f    	bl	0x100000b54 <__ZNSt27__type_info_implementations30__non_unique_arm_rtti_bit_impl21__is_type_name_uniqueB8ne200100Em>
100000adc: 370000c0    	tbnz	w0, #0x0, 0x100000af4 <__ZNKSt9type_infoeqB8ne200100ERKS_+0x74>
100000ae0: 14000001    	b	0x100000ae4 <__ZNKSt9type_infoeqB8ne200100ERKS_+0x64>
100000ae4: f9400fe0    	ldr	x0, [sp, #0x18]
100000ae8: 9400001b    	bl	0x100000b54 <__ZNSt27__type_info_implementations30__non_unique_arm_rtti_bit_impl21__is_type_name_uniqueB8ne200100Em>
100000aec: 360000e0    	tbz	w0, #0x0, 0x100000b08 <__ZNKSt9type_infoeqB8ne200100ERKS_+0x88>
100000af0: 14000001    	b	0x100000af4 <__ZNKSt9type_infoeqB8ne200100ERKS_+0x74>
100000af4: 52800008    	mov	w8, #0x0                ; =0
100000af8: 12000108    	and	w8, w8, #0x1
100000afc: 12000108    	and	w8, w8, #0x1
100000b00: 381ef3a8    	sturb	w8, [x29, #-0x11]
100000b04: 1400000f    	b	0x100000b40 <__ZNKSt9type_infoeqB8ne200100ERKS_+0xc0>
100000b08: f94013e8    	ldr	x8, [sp, #0x20]
100000b0c: f81f83a8    	stur	x8, [x29, #-0x8]
100000b10: f85f83a8    	ldur	x8, [x29, #-0x8]
100000b14: 9240f900    	and	x0, x8, #0x7fffffffffffffff
100000b18: f9400fe8    	ldr	x8, [sp, #0x18]
100000b1c: f81f03a8    	stur	x8, [x29, #-0x10]
100000b20: f85f03a8    	ldur	x8, [x29, #-0x10]
100000b24: 9240f901    	and	x1, x8, #0x7fffffffffffffff
100000b28: 940000db    	bl	0x100000e94 <_strcmp+0x100000e94>
100000b2c: 71000008    	subs	w8, w0, #0x0
100000b30: 1a9f17e8    	cset	w8, eq
100000b34: 12000108    	and	w8, w8, #0x1
100000b38: 381ef3a8    	sturb	w8, [x29, #-0x11]
100000b3c: 14000001    	b	0x100000b40 <__ZNKSt9type_infoeqB8ne200100ERKS_+0xc0>
100000b40: 385ef3a8    	ldurb	w8, [x29, #-0x11]
100000b44: 12000100    	and	w0, w8, #0x1
100000b48: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000b4c: 910143ff    	add	sp, sp, #0x50
100000b50: d65f03c0    	ret

0000000100000b54 <__ZNSt27__type_info_implementations30__non_unique_arm_rtti_bit_impl21__is_type_name_uniqueB8ne200100Em>:
100000b54: d10043ff    	sub	sp, sp, #0x10
100000b58: f90007e0    	str	x0, [sp, #0x8]
100000b5c: f94007e8    	ldr	x8, [sp, #0x8]
100000b60: f2410108    	ands	x8, x8, #0x8000000000000000
100000b64: 1a9f17e0    	cset	w0, eq
100000b68: 910043ff    	add	sp, sp, #0x10
100000b6c: d65f03c0    	ret

0000000100000b70 <__ZNSt3__19allocatorINS_20__shared_ptr_pointerIPiPFvS2_ENS0_IiEEEEEC1B8ne200100IiEERKNS0_IT_EE>:
100000b70: d100c3ff    	sub	sp, sp, #0x30
100000b74: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000b78: 910083fd    	add	x29, sp, #0x20
100000b7c: f81f83a0    	stur	x0, [x29, #-0x8]
100000b80: f9000be1    	str	x1, [sp, #0x10]
100000b84: f85f83a0    	ldur	x0, [x29, #-0x8]
100000b88: f90007e0    	str	x0, [sp, #0x8]
100000b8c: f9400be1    	ldr	x1, [sp, #0x10]
100000b90: 94000017    	bl	0x100000bec <__ZNSt3__19allocatorINS_20__shared_ptr_pointerIPiPFvS2_ENS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>
100000b94: f94007e0    	ldr	x0, [sp, #0x8]
100000b98: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000b9c: 9100c3ff    	add	sp, sp, #0x30
100000ba0: d65f03c0    	ret

0000000100000ba4 <__ZNSt3__19allocatorINS_20__shared_ptr_pointerIPiPFvS2_ENS0_IiEEEEE10deallocateB8ne200100EPS6_m>:
100000ba4: d100c3ff    	sub	sp, sp, #0x30
100000ba8: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000bac: 910083fd    	add	x29, sp, #0x20
100000bb0: f81f83a0    	stur	x0, [x29, #-0x8]
100000bb4: f9000be1    	str	x1, [sp, #0x10]
100000bb8: f90007e2    	str	x2, [sp, #0x8]
100000bbc: f9400be0    	ldr	x0, [sp, #0x10]
100000bc0: f94007e1    	ldr	x1, [sp, #0x8]
100000bc4: d2800102    	mov	x2, #0x8                ; =8
100000bc8: 9400001a    	bl	0x100000c30 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>
100000bcc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000bd0: 9100c3ff    	add	sp, sp, #0x30
100000bd4: d65f03c0    	ret

0000000100000bd8 <__ZNSt3__114pointer_traitsIPNS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEE10pointer_toB8ne200100ERS7_>:
100000bd8: d10043ff    	sub	sp, sp, #0x10
100000bdc: f90007e0    	str	x0, [sp, #0x8]
100000be0: f94007e0    	ldr	x0, [sp, #0x8]
100000be4: 910043ff    	add	sp, sp, #0x10
100000be8: d65f03c0    	ret

0000000100000bec <__ZNSt3__19allocatorINS_20__shared_ptr_pointerIPiPFvS2_ENS0_IiEEEEEC2B8ne200100IiEERKNS0_IT_EE>:
100000bec: d100c3ff    	sub	sp, sp, #0x30
100000bf0: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000bf4: 910083fd    	add	x29, sp, #0x20
100000bf8: f81f83a0    	stur	x0, [x29, #-0x8]
100000bfc: f9000be1    	str	x1, [sp, #0x10]
100000c00: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c04: f90007e0    	str	x0, [sp, #0x8]
100000c08: 94000005    	bl	0x100000c1c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_pointerIPiPFvS3_ENS1_IiEEEEEEEC2B8ne200100Ev>
100000c0c: f94007e0    	ldr	x0, [sp, #0x8]
100000c10: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000c14: 9100c3ff    	add	sp, sp, #0x30
100000c18: d65f03c0    	ret

0000000100000c1c <__ZNSt3__116__non_trivial_ifILb1ENS_9allocatorINS_20__shared_ptr_pointerIPiPFvS3_ENS1_IiEEEEEEEC2B8ne200100Ev>:
100000c1c: d10043ff    	sub	sp, sp, #0x10
100000c20: f90007e0    	str	x0, [sp, #0x8]
100000c24: f94007e0    	ldr	x0, [sp, #0x8]
100000c28: 910043ff    	add	sp, sp, #0x10
100000c2c: d65f03c0    	ret

0000000100000c30 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm>:
100000c30: d10103ff    	sub	sp, sp, #0x40
100000c34: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000c38: 9100c3fd    	add	x29, sp, #0x30
100000c3c: f81f83a0    	stur	x0, [x29, #-0x8]
100000c40: f81f03a1    	stur	x1, [x29, #-0x10]
100000c44: f9000fe2    	str	x2, [sp, #0x18]
100000c48: f85f03a8    	ldur	x8, [x29, #-0x10]
100000c4c: d2800509    	mov	x9, #0x28               ; =40
100000c50: 9b097d08    	mul	x8, x8, x9
100000c54: f9000be8    	str	x8, [sp, #0x10]
100000c58: f9400fe0    	ldr	x0, [sp, #0x18]
100000c5c: 9400000f    	bl	0x100000c98 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>
100000c60: 36000100    	tbz	w0, #0x0, 0x100000c80 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x50>
100000c64: 14000001    	b	0x100000c68 <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x38>
100000c68: f9400fe8    	ldr	x8, [sp, #0x18]
100000c6c: f90007e8    	str	x8, [sp, #0x8]
100000c70: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c74: f94007e1    	ldr	x1, [sp, #0x8]
100000c78: 9400000f    	bl	0x100000cb4 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEESt11align_val_tEEEvDpT_>
100000c7c: 14000004    	b	0x100000c8c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x5c>
100000c80: f85f83a0    	ldur	x0, [x29, #-0x8]
100000c84: 94000017    	bl	0x100000ce0 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEEvDpT_>
100000c88: 14000001    	b	0x100000c8c <__ZNSt3__119__libcpp_deallocateB8ne200100INS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEvPNS_15__type_identityIT_E4typeENS_15__element_countEm+0x5c>
100000c8c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000c90: 910103ff    	add	sp, sp, #0x40
100000c94: d65f03c0    	ret

0000000100000c98 <__ZNSt3__124__is_overaligned_for_newB8ne200100Em>:
100000c98: d10043ff    	sub	sp, sp, #0x10
100000c9c: f90007e0    	str	x0, [sp, #0x8]
100000ca0: f94007e8    	ldr	x8, [sp, #0x8]
100000ca4: f1004108    	subs	x8, x8, #0x10
100000ca8: 1a9f97e0    	cset	w0, hi
100000cac: 910043ff    	add	sp, sp, #0x10
100000cb0: d65f03c0    	ret

0000000100000cb4 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEESt11align_val_tEEEvDpT_>:
100000cb4: d10083ff    	sub	sp, sp, #0x20
100000cb8: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000cbc: 910043fd    	add	x29, sp, #0x10
100000cc0: f90007e0    	str	x0, [sp, #0x8]
100000cc4: f90003e1    	str	x1, [sp]
100000cc8: f94007e0    	ldr	x0, [sp, #0x8]
100000ccc: f94003e1    	ldr	x1, [sp]
100000cd0: 94000074    	bl	0x100000ea0 <_strcmp+0x100000ea0>
100000cd4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cd8: 910083ff    	add	sp, sp, #0x20
100000cdc: d65f03c0    	ret

0000000100000ce0 <__ZNSt3__124__libcpp_operator_deleteB8ne200100IJPNS_20__shared_ptr_pointerIPiPFvS2_ENS_9allocatorIiEEEEEEEvDpT_>:
100000ce0: d10083ff    	sub	sp, sp, #0x20
100000ce4: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000ce8: 910043fd    	add	x29, sp, #0x10
100000cec: f90007e0    	str	x0, [sp, #0x8]
100000cf0: f94007e0    	ldr	x0, [sp, #0x8]
100000cf4: 94000050    	bl	0x100000e34 <_strcmp+0x100000e34>
100000cf8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000cfc: 910083ff    	add	sp, sp, #0x20
100000d00: d65f03c0    	ret

0000000100000d04 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev>:
100000d04: d100c3ff    	sub	sp, sp, #0x30
100000d08: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000d0c: 910083fd    	add	x29, sp, #0x20
100000d10: f9000be0    	str	x0, [sp, #0x10]
100000d14: f9400be8    	ldr	x8, [sp, #0x10]
100000d18: f90007e8    	str	x8, [sp, #0x8]
100000d1c: aa0803e9    	mov	x9, x8
100000d20: f81f83a9    	stur	x9, [x29, #-0x8]
100000d24: f9400508    	ldr	x8, [x8, #0x8]
100000d28: b40000c8    	cbz	x8, 0x100000d40 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
100000d2c: 14000001    	b	0x100000d30 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x2c>
100000d30: f94007e8    	ldr	x8, [sp, #0x8]
100000d34: f9400500    	ldr	x0, [x8, #0x8]
100000d38: 94000006    	bl	0x100000d50 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>
100000d3c: 14000001    	b	0x100000d40 <__ZNSt3__110shared_ptrIiED2B8ne200100Ev+0x3c>
100000d40: f85f83a0    	ldur	x0, [x29, #-0x8]
100000d44: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000d48: 9100c3ff    	add	sp, sp, #0x30
100000d4c: d65f03c0    	ret

0000000100000d50 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev>:
100000d50: d10083ff    	sub	sp, sp, #0x20
100000d54: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000d58: 910043fd    	add	x29, sp, #0x10
100000d5c: f90007e0    	str	x0, [sp, #0x8]
100000d60: f94007e0    	ldr	x0, [sp, #0x8]
100000d64: f90003e0    	str	x0, [sp]
100000d68: 94000009    	bl	0x100000d8c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>
100000d6c: 360000a0    	tbz	w0, #0x0, 0x100000d80 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
100000d70: 14000001    	b	0x100000d74 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x24>
100000d74: f94003e0    	ldr	x0, [sp]
100000d78: 9400004d    	bl	0x100000eac <_strcmp+0x100000eac>
100000d7c: 14000001    	b	0x100000d80 <__ZNSt3__119__shared_weak_count16__release_sharedB8ne200100Ev+0x30>
100000d80: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000d84: 910083ff    	add	sp, sp, #0x20
100000d88: d65f03c0    	ret

0000000100000d8c <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev>:
100000d8c: d100c3ff    	sub	sp, sp, #0x30
100000d90: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000d94: 910083fd    	add	x29, sp, #0x20
100000d98: f9000be0    	str	x0, [sp, #0x10]
100000d9c: f9400be8    	ldr	x8, [sp, #0x10]
100000da0: f90007e8    	str	x8, [sp, #0x8]
100000da4: 91002100    	add	x0, x8, #0x8
100000da8: 94000017    	bl	0x100000e04 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>
100000dac: b1000408    	adds	x8, x0, #0x1
100000db0: 54000161    	b.ne	0x100000ddc <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x50>
100000db4: 14000001    	b	0x100000db8 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x2c>
100000db8: f94007e0    	ldr	x0, [sp, #0x8]
100000dbc: f9400008    	ldr	x8, [x0]
100000dc0: f9400908    	ldr	x8, [x8, #0x10]
100000dc4: d63f0100    	blr	x8
100000dc8: 52800028    	mov	w8, #0x1                ; =1
100000dcc: 12000108    	and	w8, w8, #0x1
100000dd0: 12000108    	and	w8, w8, #0x1
100000dd4: 381ff3a8    	sturb	w8, [x29, #-0x1]
100000dd8: 14000006    	b	0x100000df0 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100000ddc: 52800008    	mov	w8, #0x0                ; =0
100000de0: 12000108    	and	w8, w8, #0x1
100000de4: 12000108    	and	w8, w8, #0x1
100000de8: 381ff3a8    	sturb	w8, [x29, #-0x1]
100000dec: 14000001    	b	0x100000df0 <__ZNSt3__114__shared_count16__release_sharedB8ne200100Ev+0x64>
100000df0: 385ff3a8    	ldurb	w8, [x29, #-0x1]
100000df4: 12000100    	and	w0, w8, #0x1
100000df8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000dfc: 9100c3ff    	add	sp, sp, #0x30
100000e00: d65f03c0    	ret

0000000100000e04 <__ZNSt3__134__libcpp_atomic_refcount_decrementB8ne200100IlEET_RS1_>:
100000e04: d10083ff    	sub	sp, sp, #0x20
100000e08: f9000fe0    	str	x0, [sp, #0x18]
100000e0c: f9400fe8    	ldr	x8, [sp, #0x18]
100000e10: 92800009    	mov	x9, #-0x1               ; =-1
100000e14: f9000be9    	str	x9, [sp, #0x10]
100000e18: f9400be9    	ldr	x9, [sp, #0x10]
100000e1c: f8e90108    	ldaddal	x9, x8, [x8]
100000e20: 8b090108    	add	x8, x8, x9
100000e24: f90007e8    	str	x8, [sp, #0x8]
100000e28: f94007e0    	ldr	x0, [sp, #0x8]
100000e2c: 910083ff    	add	sp, sp, #0x20
100000e30: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000e34 <__stubs>:
100000e34: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e38: f9400610    	ldr	x16, [x16, #0x8]
100000e3c: d61f0200    	br	x16
100000e40: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e44: f9400a10    	ldr	x16, [x16, #0x10]
100000e48: d61f0200    	br	x16
100000e4c: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e50: f9400e10    	ldr	x16, [x16, #0x18]
100000e54: d61f0200    	br	x16
100000e58: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e5c: f9401210    	ldr	x16, [x16, #0x20]
100000e60: d61f0200    	br	x16
100000e64: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e68: f9401610    	ldr	x16, [x16, #0x28]
100000e6c: d61f0200    	br	x16
100000e70: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e74: f9401a10    	ldr	x16, [x16, #0x30]
100000e78: d61f0200    	br	x16
100000e7c: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e80: f9401e10    	ldr	x16, [x16, #0x38]
100000e84: d61f0200    	br	x16
100000e88: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e8c: f9402a10    	ldr	x16, [x16, #0x50]
100000e90: d61f0200    	br	x16
100000e94: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000e98: f9402e10    	ldr	x16, [x16, #0x58]
100000e9c: d61f0200    	br	x16
100000ea0: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000ea4: f9403210    	ldr	x16, [x16, #0x60]
100000ea8: d61f0200    	br	x16
100000eac: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000eb0: f9403610    	ldr	x16, [x16, #0x68]
100000eb4: d61f0200    	br	x16
