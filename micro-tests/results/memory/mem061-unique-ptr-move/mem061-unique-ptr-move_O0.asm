
/Users/jim/work/cppfort/micro-tests/results/memory/mem061-unique-ptr-move/mem061-unique-ptr-move_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z20test_unique_ptr_movev>:
100000448: d10103ff    	sub	sp, sp, #0x40
10000044c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000450: 9100c3fd    	add	x29, sp, #0x30
100000454: d10033a0    	sub	x0, x29, #0xc
100000458: 52800548    	mov	w8, #0x2a               ; =42
10000045c: b81f43a8    	stur	w8, [x29, #-0xc]
100000460: d10023a8    	sub	x8, x29, #0x8
100000464: f90007e8    	str	x8, [sp, #0x8]
100000468: 94000012    	bl	0x1000004b0 <__ZNSt3__111make_uniqueB8ne200100IiJiELi0EEENS_10unique_ptrIT_NS_14default_deleteIS2_EEEEDpOT0_>
10000046c: f94007e1    	ldr	x1, [sp, #0x8]
100000470: 910063e0    	add	x0, sp, #0x18
100000474: f90003e0    	str	x0, [sp]
100000478: 9400001f    	bl	0x1000004f4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC1B8ne200100EOS3_>
10000047c: f94003e0    	ldr	x0, [sp]
100000480: 9400002a    	bl	0x100000528 <__ZNKSt3__110unique_ptrIiNS_14default_deleteIiEEEdeB8ne200100Ev>
100000484: aa0003e8    	mov	x8, x0
100000488: f94003e0    	ldr	x0, [sp]
10000048c: b9400108    	ldr	w8, [x8]
100000490: b90017e8    	str	w8, [sp, #0x14]
100000494: 9400002b    	bl	0x100000540 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED1B8ne200100Ev>
100000498: f94007e0    	ldr	x0, [sp, #0x8]
10000049c: 94000029    	bl	0x100000540 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED1B8ne200100Ev>
1000004a0: b94017e0    	ldr	w0, [sp, #0x14]
1000004a4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000004a8: 910103ff    	add	sp, sp, #0x40
1000004ac: d65f03c0    	ret

00000001000004b0 <__ZNSt3__111make_uniqueB8ne200100IiJiELi0EEENS_10unique_ptrIT_NS_14default_deleteIS2_EEEEDpOT0_>:
1000004b0: d100c3ff    	sub	sp, sp, #0x30
1000004b4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004b8: 910083fd    	add	x29, sp, #0x20
1000004bc: f90007e8    	str	x8, [sp, #0x8]
1000004c0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004c4: f9000be0    	str	x0, [sp, #0x10]
1000004c8: d2800080    	mov	x0, #0x4                ; =4
1000004cc: 94000094    	bl	0x10000071c <__Znwm+0x10000071c>
1000004d0: aa0003e1    	mov	x1, x0
1000004d4: f94007e0    	ldr	x0, [sp, #0x8]
1000004d8: f9400be8    	ldr	x8, [sp, #0x10]
1000004dc: b9400108    	ldr	w8, [x8]
1000004e0: b9000028    	str	w8, [x1]
1000004e4: 9400002a    	bl	0x10000058c <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC1B8ne200100ILb1EvEEPi>
1000004e8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004ec: 9100c3ff    	add	sp, sp, #0x30
1000004f0: d65f03c0    	ret

00000001000004f4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC1B8ne200100EOS3_>:
1000004f4: d100c3ff    	sub	sp, sp, #0x30
1000004f8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004fc: 910083fd    	add	x29, sp, #0x20
100000500: f81f83a0    	stur	x0, [x29, #-0x8]
100000504: f9000be1    	str	x1, [sp, #0x10]
100000508: f85f83a0    	ldur	x0, [x29, #-0x8]
10000050c: f90007e0    	str	x0, [sp, #0x8]
100000510: f9400be1    	ldr	x1, [sp, #0x10]
100000514: 94000063    	bl	0x1000006a0 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC2B8ne200100EOS3_>
100000518: f94007e0    	ldr	x0, [sp, #0x8]
10000051c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000520: 9100c3ff    	add	sp, sp, #0x30
100000524: d65f03c0    	ret

0000000100000528 <__ZNKSt3__110unique_ptrIiNS_14default_deleteIiEEEdeB8ne200100Ev>:
100000528: d10043ff    	sub	sp, sp, #0x10
10000052c: f90007e0    	str	x0, [sp, #0x8]
100000530: f94007e8    	ldr	x8, [sp, #0x8]
100000534: f9400100    	ldr	x0, [x8]
100000538: 910043ff    	add	sp, sp, #0x10
10000053c: d65f03c0    	ret

0000000100000540 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED1B8ne200100Ev>:
100000540: d10083ff    	sub	sp, sp, #0x20
100000544: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000548: 910043fd    	add	x29, sp, #0x10
10000054c: f90007e0    	str	x0, [sp, #0x8]
100000550: f94007e0    	ldr	x0, [sp, #0x8]
100000554: f90003e0    	str	x0, [sp]
100000558: 94000022    	bl	0x1000005e0 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED2B8ne200100Ev>
10000055c: f94003e0    	ldr	x0, [sp]
100000560: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000564: 910083ff    	add	sp, sp, #0x20
100000568: d65f03c0    	ret

000000010000056c <_main>:
10000056c: d10083ff    	sub	sp, sp, #0x20
100000570: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000574: 910043fd    	add	x29, sp, #0x10
100000578: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000057c: 97ffffb3    	bl	0x100000448 <__Z20test_unique_ptr_movev>
100000580: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000584: 910083ff    	add	sp, sp, #0x20
100000588: d65f03c0    	ret

000000010000058c <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC1B8ne200100ILb1EvEEPi>:
10000058c: d100c3ff    	sub	sp, sp, #0x30
100000590: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000594: 910083fd    	add	x29, sp, #0x20
100000598: f81f83a0    	stur	x0, [x29, #-0x8]
10000059c: f9000be1    	str	x1, [sp, #0x10]
1000005a0: f85f83a0    	ldur	x0, [x29, #-0x8]
1000005a4: f90007e0    	str	x0, [sp, #0x8]
1000005a8: f9400be1    	ldr	x1, [sp, #0x10]
1000005ac: 94000005    	bl	0x1000005c0 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC2B8ne200100ILb1EvEEPi>
1000005b0: f94007e0    	ldr	x0, [sp, #0x8]
1000005b4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000005b8: 9100c3ff    	add	sp, sp, #0x30
1000005bc: d65f03c0    	ret

00000001000005c0 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC2B8ne200100ILb1EvEEPi>:
1000005c0: d10043ff    	sub	sp, sp, #0x10
1000005c4: f90007e0    	str	x0, [sp, #0x8]
1000005c8: f90003e1    	str	x1, [sp]
1000005cc: f94007e0    	ldr	x0, [sp, #0x8]
1000005d0: f94003e8    	ldr	x8, [sp]
1000005d4: f9000008    	str	x8, [x0]
1000005d8: 910043ff    	add	sp, sp, #0x10
1000005dc: d65f03c0    	ret

00000001000005e0 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED2B8ne200100Ev>:
1000005e0: d10083ff    	sub	sp, sp, #0x20
1000005e4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000005e8: 910043fd    	add	x29, sp, #0x10
1000005ec: f90007e0    	str	x0, [sp, #0x8]
1000005f0: f94007e0    	ldr	x0, [sp, #0x8]
1000005f4: f90003e0    	str	x0, [sp]
1000005f8: d2800001    	mov	x1, #0x0                ; =0
1000005fc: 94000005    	bl	0x100000610 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi>
100000600: f94003e0    	ldr	x0, [sp]
100000604: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000608: 910083ff    	add	sp, sp, #0x20
10000060c: d65f03c0    	ret

0000000100000610 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi>:
100000610: d100c3ff    	sub	sp, sp, #0x30
100000614: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000618: 910083fd    	add	x29, sp, #0x20
10000061c: f81f83a0    	stur	x0, [x29, #-0x8]
100000620: f9000be1    	str	x1, [sp, #0x10]
100000624: f85f83a9    	ldur	x9, [x29, #-0x8]
100000628: f90003e9    	str	x9, [sp]
10000062c: f9400128    	ldr	x8, [x9]
100000630: f90007e8    	str	x8, [sp, #0x8]
100000634: f9400be8    	ldr	x8, [sp, #0x10]
100000638: f9000128    	str	x8, [x9]
10000063c: f94007e8    	ldr	x8, [sp, #0x8]
100000640: b40000c8    	cbz	x8, 0x100000658 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi+0x48>
100000644: 14000001    	b	0x100000648 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi+0x38>
100000648: f94003e0    	ldr	x0, [sp]
10000064c: f94007e1    	ldr	x1, [sp, #0x8]
100000650: 94000005    	bl	0x100000664 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi>
100000654: 14000001    	b	0x100000658 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi+0x48>
100000658: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000065c: 9100c3ff    	add	sp, sp, #0x30
100000660: d65f03c0    	ret

0000000100000664 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi>:
100000664: d100c3ff    	sub	sp, sp, #0x30
100000668: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000066c: 910083fd    	add	x29, sp, #0x20
100000670: f81f83a0    	stur	x0, [x29, #-0x8]
100000674: f9000be1    	str	x1, [sp, #0x10]
100000678: f9400be8    	ldr	x8, [sp, #0x10]
10000067c: f90007e8    	str	x8, [sp, #0x8]
100000680: b40000a8    	cbz	x8, 0x100000694 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi+0x30>
100000684: 14000001    	b	0x100000688 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi+0x24>
100000688: f94007e0    	ldr	x0, [sp, #0x8]
10000068c: 94000027    	bl	0x100000728 <__Znwm+0x100000728>
100000690: 14000001    	b	0x100000694 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi+0x30>
100000694: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000698: 9100c3ff    	add	sp, sp, #0x30
10000069c: d65f03c0    	ret

00000001000006a0 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC2B8ne200100EOS3_>:
1000006a0: d100c3ff    	sub	sp, sp, #0x30
1000006a4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000006a8: 910083fd    	add	x29, sp, #0x20
1000006ac: f81f83a0    	stur	x0, [x29, #-0x8]
1000006b0: f9000be1    	str	x1, [sp, #0x10]
1000006b4: f85f83a8    	ldur	x8, [x29, #-0x8]
1000006b8: f90007e8    	str	x8, [sp, #0x8]
1000006bc: f9400be0    	ldr	x0, [sp, #0x10]
1000006c0: 94000009    	bl	0x1000006e4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE7releaseB8ne200100Ev>
1000006c4: f94007e8    	ldr	x8, [sp, #0x8]
1000006c8: f9000100    	str	x0, [x8]
1000006cc: f9400be0    	ldr	x0, [sp, #0x10]
1000006d0: 9400000e    	bl	0x100000708 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE11get_deleterB8ne200100Ev>
1000006d4: f94007e0    	ldr	x0, [sp, #0x8]
1000006d8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000006dc: 9100c3ff    	add	sp, sp, #0x30
1000006e0: d65f03c0    	ret

00000001000006e4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE7releaseB8ne200100Ev>:
1000006e4: d10043ff    	sub	sp, sp, #0x10
1000006e8: f90007e0    	str	x0, [sp, #0x8]
1000006ec: f94007e8    	ldr	x8, [sp, #0x8]
1000006f0: f9400109    	ldr	x9, [x8]
1000006f4: f90003e9    	str	x9, [sp]
1000006f8: f900011f    	str	xzr, [x8]
1000006fc: f94003e0    	ldr	x0, [sp]
100000700: 910043ff    	add	sp, sp, #0x10
100000704: d65f03c0    	ret

0000000100000708 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE11get_deleterB8ne200100Ev>:
100000708: d10043ff    	sub	sp, sp, #0x10
10000070c: f90007e0    	str	x0, [sp, #0x8]
100000710: f94007e0    	ldr	x0, [sp, #0x8]
100000714: 910043ff    	add	sp, sp, #0x10
100000718: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000071c <__stubs>:
10000071c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000720: f9400210    	ldr	x16, [x16]
100000724: d61f0200    	br	x16
100000728: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000072c: f9400610    	ldr	x16, [x16, #0x8]
100000730: d61f0200    	br	x16
