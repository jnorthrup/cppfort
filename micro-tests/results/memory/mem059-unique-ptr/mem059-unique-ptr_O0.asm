
/Users/jim/work/cppfort/micro-tests/results/memory/mem059-unique-ptr/mem059-unique-ptr_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z15test_unique_ptrv>:
100000448: d100c3ff    	sub	sp, sp, #0x30
10000044c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000450: 910083fd    	add	x29, sp, #0x20
100000454: d10033a0    	sub	x0, x29, #0xc
100000458: 52800548    	mov	w8, #0x2a               ; =42
10000045c: b81f43a8    	stur	w8, [x29, #-0xc]
100000460: d10023a8    	sub	x8, x29, #0x8
100000464: f90007e8    	str	x8, [sp, #0x8]
100000468: 9400000c    	bl	0x100000498 <__ZNSt3__111make_uniqueB8ne200100IiJiELi0EEENS_10unique_ptrIT_NS_14default_deleteIS2_EEEEDpOT0_>
10000046c: f94007e0    	ldr	x0, [sp, #0x8]
100000470: 9400001b    	bl	0x1000004dc <__ZNKSt3__110unique_ptrIiNS_14default_deleteIiEEEdeB8ne200100Ev>
100000474: aa0003e8    	mov	x8, x0
100000478: f94007e0    	ldr	x0, [sp, #0x8]
10000047c: b9400108    	ldr	w8, [x8]
100000480: b90013e8    	str	w8, [sp, #0x10]
100000484: 9400001c    	bl	0x1000004f4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED1B8ne200100Ev>
100000488: b94013e0    	ldr	w0, [sp, #0x10]
10000048c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000490: 9100c3ff    	add	sp, sp, #0x30
100000494: d65f03c0    	ret

0000000100000498 <__ZNSt3__111make_uniqueB8ne200100IiJiELi0EEENS_10unique_ptrIT_NS_14default_deleteIS2_EEEEDpOT0_>:
100000498: d100c3ff    	sub	sp, sp, #0x30
10000049c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004a0: 910083fd    	add	x29, sp, #0x20
1000004a4: f90007e8    	str	x8, [sp, #0x8]
1000004a8: f81f83a8    	stur	x8, [x29, #-0x8]
1000004ac: f9000be0    	str	x0, [sp, #0x10]
1000004b0: d2800080    	mov	x0, #0x4                ; =4
1000004b4: 94000068    	bl	0x100000654 <__Znwm+0x100000654>
1000004b8: aa0003e1    	mov	x1, x0
1000004bc: f94007e0    	ldr	x0, [sp, #0x8]
1000004c0: f9400be8    	ldr	x8, [sp, #0x10]
1000004c4: b9400108    	ldr	w8, [x8]
1000004c8: b9000028    	str	w8, [x1]
1000004cc: 9400001d    	bl	0x100000540 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC1B8ne200100ILb1EvEEPi>
1000004d0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004d4: 9100c3ff    	add	sp, sp, #0x30
1000004d8: d65f03c0    	ret

00000001000004dc <__ZNKSt3__110unique_ptrIiNS_14default_deleteIiEEEdeB8ne200100Ev>:
1000004dc: d10043ff    	sub	sp, sp, #0x10
1000004e0: f90007e0    	str	x0, [sp, #0x8]
1000004e4: f94007e8    	ldr	x8, [sp, #0x8]
1000004e8: f9400100    	ldr	x0, [x8]
1000004ec: 910043ff    	add	sp, sp, #0x10
1000004f0: d65f03c0    	ret

00000001000004f4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED1B8ne200100Ev>:
1000004f4: d10083ff    	sub	sp, sp, #0x20
1000004f8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004fc: 910043fd    	add	x29, sp, #0x10
100000500: f90007e0    	str	x0, [sp, #0x8]
100000504: f94007e0    	ldr	x0, [sp, #0x8]
100000508: f90003e0    	str	x0, [sp]
10000050c: 94000022    	bl	0x100000594 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED2B8ne200100Ev>
100000510: f94003e0    	ldr	x0, [sp]
100000514: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000518: 910083ff    	add	sp, sp, #0x20
10000051c: d65f03c0    	ret

0000000100000520 <_main>:
100000520: d10083ff    	sub	sp, sp, #0x20
100000524: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000528: 910043fd    	add	x29, sp, #0x10
10000052c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000530: 97ffffc6    	bl	0x100000448 <__Z15test_unique_ptrv>
100000534: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000538: 910083ff    	add	sp, sp, #0x20
10000053c: d65f03c0    	ret

0000000100000540 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC1B8ne200100ILb1EvEEPi>:
100000540: d100c3ff    	sub	sp, sp, #0x30
100000544: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000548: 910083fd    	add	x29, sp, #0x20
10000054c: f81f83a0    	stur	x0, [x29, #-0x8]
100000550: f9000be1    	str	x1, [sp, #0x10]
100000554: f85f83a0    	ldur	x0, [x29, #-0x8]
100000558: f90007e0    	str	x0, [sp, #0x8]
10000055c: f9400be1    	ldr	x1, [sp, #0x10]
100000560: 94000005    	bl	0x100000574 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC2B8ne200100ILb1EvEEPi>
100000564: f94007e0    	ldr	x0, [sp, #0x8]
100000568: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000056c: 9100c3ff    	add	sp, sp, #0x30
100000570: d65f03c0    	ret

0000000100000574 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEEC2B8ne200100ILb1EvEEPi>:
100000574: d10043ff    	sub	sp, sp, #0x10
100000578: f90007e0    	str	x0, [sp, #0x8]
10000057c: f90003e1    	str	x1, [sp]
100000580: f94007e0    	ldr	x0, [sp, #0x8]
100000584: f94003e8    	ldr	x8, [sp]
100000588: f9000008    	str	x8, [x0]
10000058c: 910043ff    	add	sp, sp, #0x10
100000590: d65f03c0    	ret

0000000100000594 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEED2B8ne200100Ev>:
100000594: d10083ff    	sub	sp, sp, #0x20
100000598: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000059c: 910043fd    	add	x29, sp, #0x10
1000005a0: f90007e0    	str	x0, [sp, #0x8]
1000005a4: f94007e0    	ldr	x0, [sp, #0x8]
1000005a8: f90003e0    	str	x0, [sp]
1000005ac: d2800001    	mov	x1, #0x0                ; =0
1000005b0: 94000005    	bl	0x1000005c4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi>
1000005b4: f94003e0    	ldr	x0, [sp]
1000005b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005bc: 910083ff    	add	sp, sp, #0x20
1000005c0: d65f03c0    	ret

00000001000005c4 <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi>:
1000005c4: d100c3ff    	sub	sp, sp, #0x30
1000005c8: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005cc: 910083fd    	add	x29, sp, #0x20
1000005d0: f81f83a0    	stur	x0, [x29, #-0x8]
1000005d4: f9000be1    	str	x1, [sp, #0x10]
1000005d8: f85f83a9    	ldur	x9, [x29, #-0x8]
1000005dc: f90003e9    	str	x9, [sp]
1000005e0: f9400128    	ldr	x8, [x9]
1000005e4: f90007e8    	str	x8, [sp, #0x8]
1000005e8: f9400be8    	ldr	x8, [sp, #0x10]
1000005ec: f9000128    	str	x8, [x9]
1000005f0: f94007e8    	ldr	x8, [sp, #0x8]
1000005f4: b40000c8    	cbz	x8, 0x10000060c <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi+0x48>
1000005f8: 14000001    	b	0x1000005fc <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi+0x38>
1000005fc: f94003e0    	ldr	x0, [sp]
100000600: f94007e1    	ldr	x1, [sp, #0x8]
100000604: 94000005    	bl	0x100000618 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi>
100000608: 14000001    	b	0x10000060c <__ZNSt3__110unique_ptrIiNS_14default_deleteIiEEE5resetB8ne200100EPi+0x48>
10000060c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000610: 9100c3ff    	add	sp, sp, #0x30
100000614: d65f03c0    	ret

0000000100000618 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi>:
100000618: d100c3ff    	sub	sp, sp, #0x30
10000061c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000620: 910083fd    	add	x29, sp, #0x20
100000624: f81f83a0    	stur	x0, [x29, #-0x8]
100000628: f9000be1    	str	x1, [sp, #0x10]
10000062c: f9400be8    	ldr	x8, [sp, #0x10]
100000630: f90007e8    	str	x8, [sp, #0x8]
100000634: b40000a8    	cbz	x8, 0x100000648 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi+0x30>
100000638: 14000001    	b	0x10000063c <__ZNKSt3__114default_deleteIiEclB8ne200100EPi+0x24>
10000063c: f94007e0    	ldr	x0, [sp, #0x8]
100000640: 94000008    	bl	0x100000660 <__Znwm+0x100000660>
100000644: 14000001    	b	0x100000648 <__ZNKSt3__114default_deleteIiEclB8ne200100EPi+0x30>
100000648: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000064c: 9100c3ff    	add	sp, sp, #0x30
100000650: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000654 <__stubs>:
100000654: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000658: f9400210    	ldr	x16, [x16]
10000065c: d61f0200    	br	x16
100000660: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000664: f9400610    	ldr	x16, [x16, #0x8]
100000668: d61f0200    	br	x16
