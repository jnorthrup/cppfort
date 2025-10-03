
/Users/jim/work/cppfort/micro-tests/results/memory/mem060-unique-ptr-array/mem060-unique-ptr-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z21test_unique_ptr_arrayv>:
100000498: d10143ff    	sub	sp, sp, #0x50
10000049c: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000004a0: 910103fd    	add	x29, sp, #0x40
1000004a4: d10023a8    	sub	x8, x29, #0x8
1000004a8: f9000fe8    	str	x8, [sp, #0x18]
1000004ac: d28000a0    	mov	x0, #0x5                ; =5
1000004b0: 9400001f    	bl	0x10000052c <__ZNSt3__111make_uniqueB8ne200100IA_iLi0EEENS_10unique_ptrIT_NS_14default_deleteIS3_EEEEm>
1000004b4: f9400fe0    	ldr	x0, [sp, #0x18]
1000004b8: d2800041    	mov	x1, #0x2                ; =2
1000004bc: 94000034    	bl	0x10000058c <__ZNKSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEixB8ne200100Em>
1000004c0: f90013e0    	str	x0, [sp, #0x20]
1000004c4: 14000001    	b	0x1000004c8 <__Z21test_unique_ptr_arrayv+0x30>
1000004c8: f94013e9    	ldr	x9, [sp, #0x20]
1000004cc: 52800548    	mov	w8, #0x2a               ; =42
1000004d0: b9000128    	str	w8, [x9]
1000004d4: d10023a0    	sub	x0, x29, #0x8
1000004d8: d2800041    	mov	x1, #0x2                ; =2
1000004dc: 9400002c    	bl	0x10000058c <__ZNKSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEixB8ne200100Em>
1000004e0: f9000be0    	str	x0, [sp, #0x10]
1000004e4: 14000001    	b	0x1000004e8 <__Z21test_unique_ptr_arrayv+0x50>
1000004e8: f9400be8    	ldr	x8, [sp, #0x10]
1000004ec: b9400108    	ldr	w8, [x8]
1000004f0: b9000fe8    	str	w8, [sp, #0xc]
1000004f4: d10023a0    	sub	x0, x29, #0x8
1000004f8: 9400002e    	bl	0x1000005b0 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEED1B8ne200100Ev>
1000004fc: b9400fe0    	ldr	w0, [sp, #0xc]
100000500: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000504: 910143ff    	add	sp, sp, #0x50
100000508: d65f03c0    	ret
10000050c: f81f03a0    	stur	x0, [x29, #-0x10]
100000510: aa0103e8    	mov	x8, x1
100000514: b81ec3a8    	stur	w8, [x29, #-0x14]
100000518: d10023a0    	sub	x0, x29, #0x8
10000051c: 94000025    	bl	0x1000005b0 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEED1B8ne200100Ev>
100000520: 14000001    	b	0x100000524 <__Z21test_unique_ptr_arrayv+0x8c>
100000524: f85f03a0    	ldur	x0, [x29, #-0x10]
100000528: 9400009c    	bl	0x100000798 <_bzero+0x100000798>

000000010000052c <__ZNSt3__111make_uniqueB8ne200100IA_iLi0EEENS_10unique_ptrIT_NS_14default_deleteIS3_EEEEm>:
10000052c: d10103ff    	sub	sp, sp, #0x40
100000530: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000534: 9100c3fd    	add	x29, sp, #0x30
100000538: f90007e8    	str	x8, [sp, #0x8]
10000053c: f81f83a8    	stur	x8, [x29, #-0x8]
100000540: f81f03a0    	stur	x0, [x29, #-0x10]
100000544: f85f03a8    	ldur	x8, [x29, #-0x10]
100000548: d280008a    	mov	x10, #0x4               ; =4
10000054c: 9bca7d09    	umulh	x9, x8, x10
100000550: 9b0a7d08    	mul	x8, x8, x10
100000554: f1000129    	subs	x9, x9, #0x0
100000558: da9f0100    	csinv	x0, x8, xzr, eq
10000055c: f90003e0    	str	x0, [sp]
100000560: 94000091    	bl	0x1000007a4 <_bzero+0x1000007a4>
100000564: f94003e1    	ldr	x1, [sp]
100000568: f9000be0    	str	x0, [sp, #0x10]
10000056c: 94000091    	bl	0x1000007b0 <_bzero+0x1000007b0>
100000570: f94007e0    	ldr	x0, [sp, #0x8]
100000574: f9400be1    	ldr	x1, [sp, #0x10]
100000578: f85f03a2    	ldur	x2, [x29, #-0x10]
10000057c: 94000020    	bl	0x1000005fc <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEC1B8ne200100INS_25__private_constructor_tagEPiLi0EEET_T0_m>
100000580: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000584: 910103ff    	add	sp, sp, #0x40
100000588: d65f03c0    	ret

000000010000058c <__ZNKSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEixB8ne200100Em>:
10000058c: d10043ff    	sub	sp, sp, #0x10
100000590: f90007e0    	str	x0, [sp, #0x8]
100000594: f90003e1    	str	x1, [sp]
100000598: f94007e8    	ldr	x8, [sp, #0x8]
10000059c: f9400108    	ldr	x8, [x8]
1000005a0: f94003e9    	ldr	x9, [sp]
1000005a4: 8b090900    	add	x0, x8, x9, lsl #2
1000005a8: 910043ff    	add	sp, sp, #0x10
1000005ac: d65f03c0    	ret

00000001000005b0 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEED1B8ne200100Ev>:
1000005b0: d10083ff    	sub	sp, sp, #0x20
1000005b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000005b8: 910043fd    	add	x29, sp, #0x10
1000005bc: f90007e0    	str	x0, [sp, #0x8]
1000005c0: f94007e0    	ldr	x0, [sp, #0x8]
1000005c4: f90003e0    	str	x0, [sp]
1000005c8: 94000045    	bl	0x1000006dc <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEED2B8ne200100Ev>
1000005cc: f94003e0    	ldr	x0, [sp]
1000005d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005d4: 910083ff    	add	sp, sp, #0x20
1000005d8: d65f03c0    	ret

00000001000005dc <_main>:
1000005dc: d10083ff    	sub	sp, sp, #0x20
1000005e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000005e4: 910043fd    	add	x29, sp, #0x10
1000005e8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000005ec: 97ffffab    	bl	0x100000498 <__Z21test_unique_ptr_arrayv>
1000005f0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005f4: 910083ff    	add	sp, sp, #0x20
1000005f8: d65f03c0    	ret

00000001000005fc <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEC1B8ne200100INS_25__private_constructor_tagEPiLi0EEET_T0_m>:
1000005fc: d10103ff    	sub	sp, sp, #0x40
100000600: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000604: 9100c3fd    	add	x29, sp, #0x30
100000608: f81f03a0    	stur	x0, [x29, #-0x10]
10000060c: f9000fe1    	str	x1, [sp, #0x18]
100000610: f9000be2    	str	x2, [sp, #0x10]
100000614: f85f03a0    	ldur	x0, [x29, #-0x10]
100000618: f90007e0    	str	x0, [sp, #0x8]
10000061c: f9400fe1    	ldr	x1, [sp, #0x18]
100000620: f9400be2    	ldr	x2, [sp, #0x10]
100000624: 94000005    	bl	0x100000638 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEC2B8ne200100INS_25__private_constructor_tagEPiLi0EEET_T0_m>
100000628: f94007e0    	ldr	x0, [sp, #0x8]
10000062c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000630: 910103ff    	add	sp, sp, #0x40
100000634: d65f03c0    	ret

0000000100000638 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEC2B8ne200100INS_25__private_constructor_tagEPiLi0EEET_T0_m>:
100000638: d10103ff    	sub	sp, sp, #0x40
10000063c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000640: 9100c3fd    	add	x29, sp, #0x30
100000644: f81f03a0    	stur	x0, [x29, #-0x10]
100000648: f9000fe1    	str	x1, [sp, #0x18]
10000064c: f9000be2    	str	x2, [sp, #0x10]
100000650: f85f03a0    	ldur	x0, [x29, #-0x10]
100000654: f90007e0    	str	x0, [sp, #0x8]
100000658: f9400fe8    	ldr	x8, [sp, #0x18]
10000065c: f9000008    	str	x8, [x0]
100000660: f9400be1    	ldr	x1, [sp, #0x10]
100000664: 94000007    	bl	0x100000680 <__ZNSt3__135__unique_ptr_array_bounds_statelessC1B8ne200100Em>
100000668: 14000001    	b	0x10000066c <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEEC2B8ne200100INS_25__private_constructor_tagEPiLi0EEET_T0_m+0x34>
10000066c: f94007e0    	ldr	x0, [sp, #0x8]
100000670: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000674: 910103ff    	add	sp, sp, #0x40
100000678: d65f03c0    	ret
10000067c: 9400000e    	bl	0x1000006b4 <___clang_call_terminate>

0000000100000680 <__ZNSt3__135__unique_ptr_array_bounds_statelessC1B8ne200100Em>:
100000680: d100c3ff    	sub	sp, sp, #0x30
100000684: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000688: 910083fd    	add	x29, sp, #0x20
10000068c: f81f83a0    	stur	x0, [x29, #-0x8]
100000690: f9000be1    	str	x1, [sp, #0x10]
100000694: f85f83a0    	ldur	x0, [x29, #-0x8]
100000698: f90007e0    	str	x0, [sp, #0x8]
10000069c: f9400be1    	ldr	x1, [sp, #0x10]
1000006a0: 94000009    	bl	0x1000006c4 <__ZNSt3__135__unique_ptr_array_bounds_statelessC2B8ne200100Em>
1000006a4: f94007e0    	ldr	x0, [sp, #0x8]
1000006a8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000006ac: 9100c3ff    	add	sp, sp, #0x30
1000006b0: d65f03c0    	ret

00000001000006b4 <___clang_call_terminate>:
1000006b4: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000006b8: 910003fd    	mov	x29, sp
1000006bc: 94000040    	bl	0x1000007bc <_bzero+0x1000007bc>
1000006c0: 94000042    	bl	0x1000007c8 <_bzero+0x1000007c8>

00000001000006c4 <__ZNSt3__135__unique_ptr_array_bounds_statelessC2B8ne200100Em>:
1000006c4: d10043ff    	sub	sp, sp, #0x10
1000006c8: f90007e0    	str	x0, [sp, #0x8]
1000006cc: f90003e1    	str	x1, [sp]
1000006d0: f94007e0    	ldr	x0, [sp, #0x8]
1000006d4: 910043ff    	add	sp, sp, #0x10
1000006d8: d65f03c0    	ret

00000001000006dc <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEED2B8ne200100Ev>:
1000006dc: d10083ff    	sub	sp, sp, #0x20
1000006e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000006e4: 910043fd    	add	x29, sp, #0x10
1000006e8: f90007e0    	str	x0, [sp, #0x8]
1000006ec: f94007e0    	ldr	x0, [sp, #0x8]
1000006f0: f90003e0    	str	x0, [sp]
1000006f4: d2800001    	mov	x1, #0x0                ; =0
1000006f8: 94000005    	bl	0x10000070c <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEE5resetB8ne200100EDn>
1000006fc: f94003e0    	ldr	x0, [sp]
100000700: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000704: 910083ff    	add	sp, sp, #0x20
100000708: d65f03c0    	ret

000000010000070c <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEE5resetB8ne200100EDn>:
10000070c: d10103ff    	sub	sp, sp, #0x40
100000710: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000714: 9100c3fd    	add	x29, sp, #0x30
100000718: f81f83a0    	stur	x0, [x29, #-0x8]
10000071c: f81f03a1    	stur	x1, [x29, #-0x10]
100000720: f85f83a8    	ldur	x8, [x29, #-0x8]
100000724: f90007e8    	str	x8, [sp, #0x8]
100000728: f9400109    	ldr	x9, [x8]
10000072c: f9000fe9    	str	x9, [sp, #0x18]
100000730: f900011f    	str	xzr, [x8]
100000734: f9400fe8    	ldr	x8, [sp, #0x18]
100000738: b40000c8    	cbz	x8, 0x100000750 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEE5resetB8ne200100EDn+0x44>
10000073c: 14000001    	b	0x100000740 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEE5resetB8ne200100EDn+0x34>
100000740: f94007e0    	ldr	x0, [sp, #0x8]
100000744: f9400fe1    	ldr	x1, [sp, #0x18]
100000748: 94000023    	bl	0x1000007d4 <_bzero+0x1000007d4>
10000074c: 14000001    	b	0x100000750 <__ZNSt3__110unique_ptrIA_iNS_14default_deleteIS1_EEE5resetB8ne200100EDn+0x44>
100000750: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000754: 910103ff    	add	sp, sp, #0x40
100000758: d65f03c0    	ret

000000010000075c <__ZNKSt3__114default_deleteIA_iEclB8ne200100IiEENS2_20_EnableIfConvertibleIT_E4typeEPS5_>:
10000075c: d100c3ff    	sub	sp, sp, #0x30
100000760: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000764: 910083fd    	add	x29, sp, #0x20
100000768: f81f83a0    	stur	x0, [x29, #-0x8]
10000076c: f9000be1    	str	x1, [sp, #0x10]
100000770: f9400be8    	ldr	x8, [sp, #0x10]
100000774: f90007e8    	str	x8, [sp, #0x8]
100000778: b40000a8    	cbz	x8, 0x10000078c <__ZNKSt3__114default_deleteIA_iEclB8ne200100IiEENS2_20_EnableIfConvertibleIT_E4typeEPS5_+0x30>
10000077c: 14000001    	b	0x100000780 <__ZNKSt3__114default_deleteIA_iEclB8ne200100IiEENS2_20_EnableIfConvertibleIT_E4typeEPS5_+0x24>
100000780: f94007e0    	ldr	x0, [sp, #0x8]
100000784: 94000017    	bl	0x1000007e0 <_bzero+0x1000007e0>
100000788: 14000001    	b	0x10000078c <__ZNKSt3__114default_deleteIA_iEclB8ne200100IiEENS2_20_EnableIfConvertibleIT_E4typeEPS5_+0x30>
10000078c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000790: 9100c3ff    	add	sp, sp, #0x30
100000794: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000798 <__stubs>:
100000798: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
10000079c: f9400610    	ldr	x16, [x16, #0x8]
1000007a0: d61f0200    	br	x16
1000007a4: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
1000007a8: f9400a10    	ldr	x16, [x16, #0x10]
1000007ac: d61f0200    	br	x16
1000007b0: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
1000007b4: f9400e10    	ldr	x16, [x16, #0x18]
1000007b8: d61f0200    	br	x16
1000007bc: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
1000007c0: f9401210    	ldr	x16, [x16, #0x20]
1000007c4: d61f0200    	br	x16
1000007c8: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
1000007cc: f9401610    	ldr	x16, [x16, #0x28]
1000007d0: d61f0200    	br	x16
1000007d4: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
1000007d8: f9401a10    	ldr	x16, [x16, #0x30]
1000007dc: d61f0200    	br	x16
1000007e0: 90000030    	adrp	x16, 0x100004000 <_bzero+0x100004000>
1000007e4: f9401e10    	ldr	x16, [x16, #0x38]
1000007e8: d61f0200    	br	x16
