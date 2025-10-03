
/Users/jim/work/cppfort/micro-tests/results/memory/mem063-shared-ptr-copy/mem063-shared-ptr-copy_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z20test_shared_ptr_copyv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 94000072    	bl	0x1000006c4 <__Znwm+0x1000006c4>
100000500: aa0003f3    	mov	x19, x0
100000504: aa0003f4    	mov	x20, x0
100000508: f8008e9f    	str	xzr, [x20, #0x8]!
10000050c: f900081f    	str	xzr, [x0, #0x10]
100000510: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000514: 91008108    	add	x8, x8, #0x20
100000518: 91004108    	add	x8, x8, #0x10
10000051c: f9000008    	str	x8, [x0]
100000520: 52800548    	mov	w8, #0x2a               ; =42
100000524: b9001808    	str	w8, [x0, #0x18]
100000528: 52800028    	mov	w8, #0x1                ; =1
10000052c: f8280288    	ldadd	x8, x8, [x20]
100000530: 92800015    	mov	x21, #-0x1              ; =-1
100000534: f8f50288    	ldaddal	x21, x8, [x20]
100000538: b4000108    	cbz	x8, 0x100000558 <__Z20test_shared_ptr_copyv+0x70>
10000053c: f8f50288    	ldaddal	x21, x8, [x20]
100000540: b40001c8    	cbz	x8, 0x100000578 <__Z20test_shared_ptr_copyv+0x90>
100000544: 52800540    	mov	w0, #0x2a               ; =42
100000548: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000054c: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000550: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000554: d65f03c0    	ret
100000558: f9400268    	ldr	x8, [x19]
10000055c: f9400908    	ldr	x8, [x8, #0x10]
100000560: aa1303e0    	mov	x0, x19
100000564: d63f0100    	blr	x8
100000568: aa1303e0    	mov	x0, x19
10000056c: 9400004d    	bl	0x1000006a0 <__Znwm+0x1000006a0>
100000570: f8f50288    	ldaddal	x21, x8, [x20]
100000574: b5fffe88    	cbnz	x8, 0x100000544 <__Z20test_shared_ptr_copyv+0x5c>
100000578: f9400268    	ldr	x8, [x19]
10000057c: f9400908    	ldr	x8, [x8, #0x10]
100000580: aa1303e0    	mov	x0, x19
100000584: d63f0100    	blr	x8
100000588: aa1303e0    	mov	x0, x19
10000058c: 94000045    	bl	0x1000006a0 <__Znwm+0x1000006a0>
100000590: 52800540    	mov	w0, #0x2a               ; =42
100000594: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000598: a9414ff4    	ldp	x20, x19, [sp, #0x10]
10000059c: a8c357f6    	ldp	x22, x21, [sp], #0x30
1000005a0: d65f03c0    	ret

00000001000005a4 <_main>:
1000005a4: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000005a8: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000005ac: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005b0: 910083fd    	add	x29, sp, #0x20
1000005b4: 52800400    	mov	w0, #0x20               ; =32
1000005b8: 94000043    	bl	0x1000006c4 <__Znwm+0x1000006c4>
1000005bc: aa0003f3    	mov	x19, x0
1000005c0: aa0003f4    	mov	x20, x0
1000005c4: f8008e9f    	str	xzr, [x20, #0x8]!
1000005c8: f900081f    	str	xzr, [x0, #0x10]
1000005cc: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005d0: 91008108    	add	x8, x8, #0x20
1000005d4: 91004108    	add	x8, x8, #0x10
1000005d8: f9000008    	str	x8, [x0]
1000005dc: 52800548    	mov	w8, #0x2a               ; =42
1000005e0: b9001808    	str	w8, [x0, #0x18]
1000005e4: 52800028    	mov	w8, #0x1                ; =1
1000005e8: f8280288    	ldadd	x8, x8, [x20]
1000005ec: 92800015    	mov	x21, #-0x1              ; =-1
1000005f0: f8f50288    	ldaddal	x21, x8, [x20]
1000005f4: b4000108    	cbz	x8, 0x100000614 <_main+0x70>
1000005f8: f8f50288    	ldaddal	x21, x8, [x20]
1000005fc: b40001c8    	cbz	x8, 0x100000634 <_main+0x90>
100000600: 52800540    	mov	w0, #0x2a               ; =42
100000604: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000608: a9414ff4    	ldp	x20, x19, [sp, #0x10]
10000060c: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000610: d65f03c0    	ret
100000614: f9400268    	ldr	x8, [x19]
100000618: f9400908    	ldr	x8, [x8, #0x10]
10000061c: aa1303e0    	mov	x0, x19
100000620: d63f0100    	blr	x8
100000624: aa1303e0    	mov	x0, x19
100000628: 9400001e    	bl	0x1000006a0 <__Znwm+0x1000006a0>
10000062c: f8f50288    	ldaddal	x21, x8, [x20]
100000630: b5fffe88    	cbnz	x8, 0x100000600 <_main+0x5c>
100000634: f9400268    	ldr	x8, [x19]
100000638: f9400908    	ldr	x8, [x8, #0x10]
10000063c: aa1303e0    	mov	x0, x19
100000640: d63f0100    	blr	x8
100000644: aa1303e0    	mov	x0, x19
100000648: 94000016    	bl	0x1000006a0 <__Znwm+0x1000006a0>
10000064c: 52800540    	mov	w0, #0x2a               ; =42
100000650: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000654: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000658: a8c357f6    	ldp	x22, x21, [sp], #0x30
10000065c: d65f03c0    	ret

0000000100000660 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000660: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000664: 91008108    	add	x8, x8, #0x20
100000668: 91004108    	add	x8, x8, #0x10
10000066c: f9000008    	str	x8, [x0]
100000670: 1400000f    	b	0x1000006ac <__Znwm+0x1000006ac>

0000000100000674 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000674: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000678: 910003fd    	mov	x29, sp
10000067c: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000680: 91008108    	add	x8, x8, #0x20
100000684: 91004108    	add	x8, x8, #0x10
100000688: f9000008    	str	x8, [x0]
10000068c: 94000008    	bl	0x1000006ac <__Znwm+0x1000006ac>
100000690: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000694: 14000009    	b	0x1000006b8 <__Znwm+0x1000006b8>

0000000100000698 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000698: d65f03c0    	ret

000000010000069c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
10000069c: 14000007    	b	0x1000006b8 <__Znwm+0x1000006b8>

Disassembly of section __TEXT,__stubs:

00000001000006a0 <__stubs>:
1000006a0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006a4: f9400210    	ldr	x16, [x16]
1000006a8: d61f0200    	br	x16
1000006ac: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006b0: f9400610    	ldr	x16, [x16, #0x8]
1000006b4: d61f0200    	br	x16
1000006b8: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006bc: f9400a10    	ldr	x16, [x16, #0x10]
1000006c0: d61f0200    	br	x16
1000006c4: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006c8: f9400e10    	ldr	x16, [x16, #0x18]
1000006cc: d61f0200    	br	x16
