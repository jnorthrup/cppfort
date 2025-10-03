
/Users/jim/work/cppfort/micro-tests/results/memory/mem062-shared-ptr/mem062-shared-ptr_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z15test_shared_ptrv>:
1000004e8: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004f0: 910043fd    	add	x29, sp, #0x10
1000004f4: 52800400    	mov	w0, #0x20               ; =32
1000004f8: 94000053    	bl	0x100000644 <__Znwm+0x100000644>
1000004fc: f900081f    	str	xzr, [x0, #0x10]
100000500: aa0003e8    	mov	x8, x0
100000504: f8008d1f    	str	xzr, [x8, #0x8]!
100000508: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
10000050c: 91008129    	add	x9, x9, #0x20
100000510: 91004129    	add	x9, x9, #0x10
100000514: f9000009    	str	x9, [x0]
100000518: 52800549    	mov	w9, #0x2a               ; =42
10000051c: b9001809    	str	w9, [x0, #0x18]
100000520: 92800009    	mov	x9, #-0x1               ; =-1
100000524: f8e90108    	ldaddal	x9, x8, [x8]
100000528: b40000a8    	cbz	x8, 0x10000053c <__Z15test_shared_ptrv+0x54>
10000052c: 52800540    	mov	w0, #0x2a               ; =42
100000530: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000534: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000538: d65f03c0    	ret
10000053c: f9400008    	ldr	x8, [x0]
100000540: f9400908    	ldr	x8, [x8, #0x10]
100000544: aa0003f3    	mov	x19, x0
100000548: d63f0100    	blr	x8
10000054c: aa1303e0    	mov	x0, x19
100000550: 94000034    	bl	0x100000620 <__Znwm+0x100000620>
100000554: 52800540    	mov	w0, #0x2a               ; =42
100000558: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000055c: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000560: d65f03c0    	ret

0000000100000564 <_main>:
100000564: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000568: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000056c: 910043fd    	add	x29, sp, #0x10
100000570: 52800400    	mov	w0, #0x20               ; =32
100000574: 94000034    	bl	0x100000644 <__Znwm+0x100000644>
100000578: f900081f    	str	xzr, [x0, #0x10]
10000057c: aa0003e8    	mov	x8, x0
100000580: f8008d1f    	str	xzr, [x8, #0x8]!
100000584: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000588: 91008129    	add	x9, x9, #0x20
10000058c: 91004129    	add	x9, x9, #0x10
100000590: f9000009    	str	x9, [x0]
100000594: 52800549    	mov	w9, #0x2a               ; =42
100000598: b9001809    	str	w9, [x0, #0x18]
10000059c: 92800009    	mov	x9, #-0x1               ; =-1
1000005a0: f8e90108    	ldaddal	x9, x8, [x8]
1000005a4: b40000a8    	cbz	x8, 0x1000005b8 <_main+0x54>
1000005a8: 52800540    	mov	w0, #0x2a               ; =42
1000005ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005b0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005b4: d65f03c0    	ret
1000005b8: f9400008    	ldr	x8, [x0]
1000005bc: f9400908    	ldr	x8, [x8, #0x10]
1000005c0: aa0003f3    	mov	x19, x0
1000005c4: d63f0100    	blr	x8
1000005c8: aa1303e0    	mov	x0, x19
1000005cc: 94000015    	bl	0x100000620 <__Znwm+0x100000620>
1000005d0: 52800540    	mov	w0, #0x2a               ; =42
1000005d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005d8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005dc: d65f03c0    	ret

00000001000005e0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
1000005e0: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005e4: 91008108    	add	x8, x8, #0x20
1000005e8: 91004108    	add	x8, x8, #0x10
1000005ec: f9000008    	str	x8, [x0]
1000005f0: 1400000f    	b	0x10000062c <__Znwm+0x10000062c>

00000001000005f4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
1000005f4: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000005f8: 910003fd    	mov	x29, sp
1000005fc: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000600: 91008108    	add	x8, x8, #0x20
100000604: 91004108    	add	x8, x8, #0x10
100000608: f9000008    	str	x8, [x0]
10000060c: 94000008    	bl	0x10000062c <__Znwm+0x10000062c>
100000610: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000614: 14000009    	b	0x100000638 <__Znwm+0x100000638>

0000000100000618 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000618: d65f03c0    	ret

000000010000061c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
10000061c: 14000007    	b	0x100000638 <__Znwm+0x100000638>

Disassembly of section __TEXT,__stubs:

0000000100000620 <__stubs>:
100000620: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000624: f9400210    	ldr	x16, [x16]
100000628: d61f0200    	br	x16
10000062c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000630: f9400610    	ldr	x16, [x16, #0x8]
100000634: d61f0200    	br	x16
100000638: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000063c: f9400a10    	ldr	x16, [x16, #0x10]
100000640: d61f0200    	br	x16
100000644: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000648: f9400e10    	ldr	x16, [x16, #0x18]
10000064c: d61f0200    	br	x16
