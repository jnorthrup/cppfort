
/Users/jim/work/cppfort/micro-tests/results/memory/mem062-shared-ptr/mem062-shared-ptr_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z15test_shared_ptrv>:
1000004e8: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004f0: 910043fd    	add	x29, sp, #0x10
1000004f4: 52800400    	mov	w0, #0x20               ; =32
1000004f8: 9400004b    	bl	0x100000624 <__Znwm+0x100000624>
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
100000528: b50000e8    	cbnz	x8, 0x100000544 <__Z15test_shared_ptrv+0x5c>
10000052c: f9400008    	ldr	x8, [x0]
100000530: f9400908    	ldr	x8, [x8, #0x10]
100000534: aa0003f3    	mov	x19, x0
100000538: d63f0100    	blr	x8
10000053c: aa1303e0    	mov	x0, x19
100000540: 94000030    	bl	0x100000600 <__Znwm+0x100000600>
100000544: 52800540    	mov	w0, #0x2a               ; =42
100000548: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000054c: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000550: d65f03c0    	ret

0000000100000554 <_main>:
100000554: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000558: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000055c: 910043fd    	add	x29, sp, #0x10
100000560: 52800400    	mov	w0, #0x20               ; =32
100000564: 94000030    	bl	0x100000624 <__Znwm+0x100000624>
100000568: f900081f    	str	xzr, [x0, #0x10]
10000056c: aa0003e8    	mov	x8, x0
100000570: f8008d1f    	str	xzr, [x8, #0x8]!
100000574: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000578: 91008129    	add	x9, x9, #0x20
10000057c: 91004129    	add	x9, x9, #0x10
100000580: f9000009    	str	x9, [x0]
100000584: 52800549    	mov	w9, #0x2a               ; =42
100000588: b9001809    	str	w9, [x0, #0x18]
10000058c: 92800009    	mov	x9, #-0x1               ; =-1
100000590: f8e90108    	ldaddal	x9, x8, [x8]
100000594: b50000e8    	cbnz	x8, 0x1000005b0 <_main+0x5c>
100000598: f9400008    	ldr	x8, [x0]
10000059c: f9400908    	ldr	x8, [x8, #0x10]
1000005a0: aa0003f3    	mov	x19, x0
1000005a4: d63f0100    	blr	x8
1000005a8: aa1303e0    	mov	x0, x19
1000005ac: 94000015    	bl	0x100000600 <__Znwm+0x100000600>
1000005b0: 52800540    	mov	w0, #0x2a               ; =42
1000005b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005b8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005bc: d65f03c0    	ret

00000001000005c0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
1000005c0: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005c4: 91008108    	add	x8, x8, #0x20
1000005c8: 91004108    	add	x8, x8, #0x10
1000005cc: f9000008    	str	x8, [x0]
1000005d0: 1400000f    	b	0x10000060c <__Znwm+0x10000060c>

00000001000005d4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
1000005d4: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000005d8: 910003fd    	mov	x29, sp
1000005dc: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005e0: 91008108    	add	x8, x8, #0x20
1000005e4: 91004108    	add	x8, x8, #0x10
1000005e8: f9000008    	str	x8, [x0]
1000005ec: 94000008    	bl	0x10000060c <__Znwm+0x10000060c>
1000005f0: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000005f4: 14000009    	b	0x100000618 <__Znwm+0x100000618>

00000001000005f8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
1000005f8: d65f03c0    	ret

00000001000005fc <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
1000005fc: 14000007    	b	0x100000618 <__Znwm+0x100000618>

Disassembly of section __TEXT,__stubs:

0000000100000600 <__stubs>:
100000600: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000604: f9400210    	ldr	x16, [x16]
100000608: d61f0200    	br	x16
10000060c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000610: f9400610    	ldr	x16, [x16, #0x8]
100000614: d61f0200    	br	x16
100000618: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000061c: f9400a10    	ldr	x16, [x16, #0x10]
100000620: d61f0200    	br	x16
100000624: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000628: f9400e10    	ldr	x16, [x16, #0x18]
10000062c: d61f0200    	br	x16
