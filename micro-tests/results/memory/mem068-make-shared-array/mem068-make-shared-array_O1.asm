
/Users/jim/work/cppfort/micro-tests/results/memory/mem068-make-shared-array/mem068-make-shared-array_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z22test_make_shared_arrayv>:
1000004e8: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004f0: 910043fd    	add	x29, sp, #0x10
1000004f4: 52800700    	mov	w0, #0x38               ; =56
1000004f8: 94000049    	bl	0x10000061c <__Znwm+0x10000061c>
1000004fc: aa0003e8    	mov	x8, x0
100000500: f8008d1f    	str	xzr, [x8, #0x8]!
100000504: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000508: 91008129    	add	x9, x9, #0x20
10000050c: 91004129    	add	x9, x9, #0x10
100000510: f9000009    	str	x9, [x0]
100000514: 528000a9    	mov	w9, #0x5                ; =5
100000518: a901241f    	stp	xzr, x9, [x0, #0x10]
10000051c: a9027c1f    	stp	xzr, xzr, [x0, #0x20]
100000520: b900301f    	str	wzr, [x0, #0x30]
100000524: 52800549    	mov	w9, #0x2a               ; =42
100000528: b9002809    	str	w9, [x0, #0x28]
10000052c: 92800009    	mov	x9, #-0x1               ; =-1
100000530: f8e90108    	ldaddal	x9, x8, [x8]
100000534: b50000e8    	cbnz	x8, 0x100000550 <__Z22test_make_shared_arrayv+0x68>
100000538: f9400008    	ldr	x8, [x0]
10000053c: f9400908    	ldr	x8, [x8, #0x10]
100000540: aa0003f3    	mov	x19, x0
100000544: d63f0100    	blr	x8
100000548: aa1303e0    	mov	x0, x19
10000054c: 9400002b    	bl	0x1000005f8 <__Znwm+0x1000005f8>
100000550: 52800540    	mov	w0, #0x2a               ; =42
100000554: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000558: a8c24ff4    	ldp	x20, x19, [sp], #0x20
10000055c: d65f03c0    	ret

0000000100000560 <_main>:
100000560: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000564: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000568: 910043fd    	add	x29, sp, #0x10
10000056c: 52800700    	mov	w0, #0x38               ; =56
100000570: 9400002b    	bl	0x10000061c <__Znwm+0x10000061c>
100000574: aa0003e8    	mov	x8, x0
100000578: f8008d1f    	str	xzr, [x8, #0x8]!
10000057c: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000580: 91008129    	add	x9, x9, #0x20
100000584: 91004129    	add	x9, x9, #0x10
100000588: f9000009    	str	x9, [x0]
10000058c: 528000a9    	mov	w9, #0x5                ; =5
100000590: a901241f    	stp	xzr, x9, [x0, #0x10]
100000594: a9027c1f    	stp	xzr, xzr, [x0, #0x20]
100000598: b900301f    	str	wzr, [x0, #0x30]
10000059c: 52800549    	mov	w9, #0x2a               ; =42
1000005a0: b9002809    	str	w9, [x0, #0x28]
1000005a4: 92800009    	mov	x9, #-0x1               ; =-1
1000005a8: f8e90108    	ldaddal	x9, x8, [x8]
1000005ac: b50000e8    	cbnz	x8, 0x1000005c8 <_main+0x68>
1000005b0: f9400008    	ldr	x8, [x0]
1000005b4: f9400908    	ldr	x8, [x8, #0x10]
1000005b8: aa0003f3    	mov	x19, x0
1000005bc: d63f0100    	blr	x8
1000005c0: aa1303e0    	mov	x0, x19
1000005c4: 9400000d    	bl	0x1000005f8 <__Znwm+0x1000005f8>
1000005c8: 52800540    	mov	w0, #0x2a               ; =42
1000005cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005d0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005d4: d65f03c0    	ret

00000001000005d8 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED1Ev>:
1000005d8: 1400000b    	b	0x100000604 <__Znwm+0x100000604>

00000001000005dc <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED0Ev>:
1000005dc: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000005e0: 910003fd    	mov	x29, sp
1000005e4: 94000008    	bl	0x100000604 <__Znwm+0x100000604>
1000005e8: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000005ec: 14000009    	b	0x100000610 <__Znwm+0x100000610>

00000001000005f0 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
1000005f0: d65f03c0    	ret

00000001000005f4 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
1000005f4: 14000007    	b	0x100000610 <__Znwm+0x100000610>

Disassembly of section __TEXT,__stubs:

00000001000005f8 <__stubs>:
1000005f8: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000005fc: f9400210    	ldr	x16, [x16]
100000600: d61f0200    	br	x16
100000604: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000608: f9400610    	ldr	x16, [x16, #0x8]
10000060c: d61f0200    	br	x16
100000610: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000614: f9400a10    	ldr	x16, [x16, #0x10]
100000618: d61f0200    	br	x16
10000061c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000620: f9400e10    	ldr	x16, [x16, #0x18]
100000624: d61f0200    	br	x16
