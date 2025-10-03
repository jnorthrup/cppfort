
/Users/jim/work/cppfort/micro-tests/results/memory/mem068-make-shared-array/mem068-make-shared-array_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z22test_make_shared_arrayv>:
1000004e8: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004f0: 910043fd    	add	x29, sp, #0x10
1000004f4: 52800700    	mov	w0, #0x38               ; =56
1000004f8: 94000051    	bl	0x10000063c <__Znwm+0x10000063c>
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
100000534: b40000a8    	cbz	x8, 0x100000548 <__Z22test_make_shared_arrayv+0x60>
100000538: 52800540    	mov	w0, #0x2a               ; =42
10000053c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000540: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000544: d65f03c0    	ret
100000548: f9400008    	ldr	x8, [x0]
10000054c: f9400908    	ldr	x8, [x8, #0x10]
100000550: aa0003f3    	mov	x19, x0
100000554: d63f0100    	blr	x8
100000558: aa1303e0    	mov	x0, x19
10000055c: 9400002f    	bl	0x100000618 <__Znwm+0x100000618>
100000560: 52800540    	mov	w0, #0x2a               ; =42
100000564: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000568: a8c24ff4    	ldp	x20, x19, [sp], #0x20
10000056c: d65f03c0    	ret

0000000100000570 <_main>:
100000570: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000574: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000578: 910043fd    	add	x29, sp, #0x10
10000057c: 52800700    	mov	w0, #0x38               ; =56
100000580: 9400002f    	bl	0x10000063c <__Znwm+0x10000063c>
100000584: aa0003e8    	mov	x8, x0
100000588: f8008d1f    	str	xzr, [x8, #0x8]!
10000058c: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000590: 91008129    	add	x9, x9, #0x20
100000594: 91004129    	add	x9, x9, #0x10
100000598: f9000009    	str	x9, [x0]
10000059c: 528000a9    	mov	w9, #0x5                ; =5
1000005a0: a901241f    	stp	xzr, x9, [x0, #0x10]
1000005a4: a9027c1f    	stp	xzr, xzr, [x0, #0x20]
1000005a8: b900301f    	str	wzr, [x0, #0x30]
1000005ac: 52800549    	mov	w9, #0x2a               ; =42
1000005b0: b9002809    	str	w9, [x0, #0x28]
1000005b4: 92800009    	mov	x9, #-0x1               ; =-1
1000005b8: f8e90108    	ldaddal	x9, x8, [x8]
1000005bc: b40000a8    	cbz	x8, 0x1000005d0 <_main+0x60>
1000005c0: 52800540    	mov	w0, #0x2a               ; =42
1000005c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005c8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005cc: d65f03c0    	ret
1000005d0: f9400008    	ldr	x8, [x0]
1000005d4: f9400908    	ldr	x8, [x8, #0x10]
1000005d8: aa0003f3    	mov	x19, x0
1000005dc: d63f0100    	blr	x8
1000005e0: aa1303e0    	mov	x0, x19
1000005e4: 9400000d    	bl	0x100000618 <__Znwm+0x100000618>
1000005e8: 52800540    	mov	w0, #0x2a               ; =42
1000005ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005f0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005f4: d65f03c0    	ret

00000001000005f8 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED1Ev>:
1000005f8: 1400000b    	b	0x100000624 <__Znwm+0x100000624>

00000001000005fc <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEED0Ev>:
1000005fc: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000600: 910003fd    	mov	x29, sp
100000604: 94000008    	bl	0x100000624 <__Znwm+0x100000624>
100000608: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000060c: 14000009    	b	0x100000630 <__Znwm+0x100000630>

0000000100000610 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
100000610: d65f03c0    	ret

0000000100000614 <__ZNSt3__131__unbounded_array_control_blockIA_iNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
100000614: 14000007    	b	0x100000630 <__Znwm+0x100000630>

Disassembly of section __TEXT,__stubs:

0000000100000618 <__stubs>:
100000618: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000061c: f9400210    	ldr	x16, [x16]
100000620: d61f0200    	br	x16
100000624: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000628: f9400610    	ldr	x16, [x16, #0x8]
10000062c: d61f0200    	br	x16
100000630: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000634: f9400a10    	ldr	x16, [x16, #0x10]
100000638: d61f0200    	br	x16
10000063c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000640: f9400e10    	ldr	x16, [x16, #0x18]
100000644: d61f0200    	br	x16
