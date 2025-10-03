
/Users/jim/work/cppfort/micro-tests/results/memory/mem065-weak-ptr/mem065-weak-ptr_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z13test_weak_ptrv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 94000049    	bl	0x100000620 <__Znwm+0x100000620>
100000500: aa0003f3    	mov	x19, x0
100000504: aa0003e8    	mov	x8, x0
100000508: f8010d1f    	str	xzr, [x8, #0x10]!
10000050c: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000510: 9100a129    	add	x9, x9, #0x28
100000514: 91004129    	add	x9, x9, #0x10
100000518: a9007c09    	stp	x9, xzr, [x0]
10000051c: 52800549    	mov	w9, #0x2a               ; =42
100000520: b9001809    	str	w9, [x0, #0x18]
100000524: 52800029    	mov	w9, #0x1                ; =1
100000528: f8290108    	ldadd	x9, x8, [x8]
10000052c: 94000034    	bl	0x1000005fc <__Znwm+0x1000005fc>
100000530: b40001a0    	cbz	x0, 0x100000564 <__Z13test_weak_ptrv+0x7c>
100000534: b9401a74    	ldr	w20, [x19, #0x18]
100000538: 91002008    	add	x8, x0, #0x8
10000053c: 92800009    	mov	x9, #-0x1               ; =-1
100000540: f8e90108    	ldaddal	x9, x8, [x8]
100000544: b5000128    	cbnz	x8, 0x100000568 <__Z13test_weak_ptrv+0x80>
100000548: f9400008    	ldr	x8, [x0]
10000054c: f9400908    	ldr	x8, [x8, #0x10]
100000550: aa0003f5    	mov	x21, x0
100000554: d63f0100    	blr	x8
100000558: aa1503e0    	mov	x0, x21
10000055c: 94000025    	bl	0x1000005f0 <__Znwm+0x1000005f0>
100000560: 14000002    	b	0x100000568 <__Z13test_weak_ptrv+0x80>
100000564: 12800014    	mov	w20, #-0x1              ; =-1
100000568: aa1303e0    	mov	x0, x19
10000056c: 94000021    	bl	0x1000005f0 <__Znwm+0x1000005f0>
100000570: 91002268    	add	x8, x19, #0x8
100000574: 92800009    	mov	x9, #-0x1               ; =-1
100000578: f8e90108    	ldaddal	x9, x8, [x8]
10000057c: b50000e8    	cbnz	x8, 0x100000598 <__Z13test_weak_ptrv+0xb0>
100000580: f9400268    	ldr	x8, [x19]
100000584: f9400908    	ldr	x8, [x8, #0x10]
100000588: aa1303e0    	mov	x0, x19
10000058c: d63f0100    	blr	x8
100000590: aa1303e0    	mov	x0, x19
100000594: 94000017    	bl	0x1000005f0 <__Znwm+0x1000005f0>
100000598: aa1403e0    	mov	x0, x20
10000059c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000005a0: a9414ff4    	ldp	x20, x19, [sp, #0x10]
1000005a4: a8c357f6    	ldp	x22, x21, [sp], #0x30
1000005a8: d65f03c0    	ret

00000001000005ac <_main>:
1000005ac: 17ffffcf    	b	0x1000004e8 <__Z13test_weak_ptrv>

00000001000005b0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
1000005b0: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005b4: 9100a108    	add	x8, x8, #0x28
1000005b8: 91004108    	add	x8, x8, #0x10
1000005bc: f9000008    	str	x8, [x0]
1000005c0: 14000012    	b	0x100000608 <__Znwm+0x100000608>

00000001000005c4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
1000005c4: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000005c8: 910003fd    	mov	x29, sp
1000005cc: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005d0: 9100a108    	add	x8, x8, #0x28
1000005d4: 91004108    	add	x8, x8, #0x10
1000005d8: f9000008    	str	x8, [x0]
1000005dc: 9400000b    	bl	0x100000608 <__Znwm+0x100000608>
1000005e0: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000005e4: 1400000c    	b	0x100000614 <__Znwm+0x100000614>

00000001000005e8 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
1000005e8: d65f03c0    	ret

00000001000005ec <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
1000005ec: 1400000a    	b	0x100000614 <__Znwm+0x100000614>

Disassembly of section __TEXT,__stubs:

00000001000005f0 <__stubs>:
1000005f0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000005f4: f9400210    	ldr	x16, [x16]
1000005f8: d61f0200    	br	x16
1000005fc: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000600: f9400610    	ldr	x16, [x16, #0x8]
100000604: d61f0200    	br	x16
100000608: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000060c: f9400a10    	ldr	x16, [x16, #0x10]
100000610: d61f0200    	br	x16
100000614: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000618: f9400e10    	ldr	x16, [x16, #0x18]
10000061c: d61f0200    	br	x16
100000620: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000624: f9401210    	ldr	x16, [x16, #0x20]
100000628: d61f0200    	br	x16
