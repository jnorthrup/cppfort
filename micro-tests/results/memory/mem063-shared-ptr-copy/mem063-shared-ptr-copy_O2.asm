
/Users/jim/work/cppfort/micro-tests/results/memory/mem063-shared-ptr-copy/mem063-shared-ptr-copy_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z20test_shared_ptr_copyv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 9400006a    	bl	0x1000006a4 <__Znwm+0x1000006a4>
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
10000056c: 94000045    	bl	0x100000680 <__Znwm+0x100000680>
100000570: f8f50288    	ldaddal	x21, x8, [x20]
100000574: b5fffe88    	cbnz	x8, 0x100000544 <__Z20test_shared_ptr_copyv+0x5c>
100000578: f9400268    	ldr	x8, [x19]
10000057c: f9400908    	ldr	x8, [x8, #0x10]
100000580: aa1303e0    	mov	x0, x19
100000584: d63f0100    	blr	x8
100000588: aa1303e0    	mov	x0, x19
10000058c: 9400003d    	bl	0x100000680 <__Znwm+0x100000680>
100000590: 17ffffed    	b	0x100000544 <__Z20test_shared_ptr_copyv+0x5c>

0000000100000594 <_main>:
100000594: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
100000598: a9014ff4    	stp	x20, x19, [sp, #0x10]
10000059c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005a0: 910083fd    	add	x29, sp, #0x20
1000005a4: 52800400    	mov	w0, #0x20               ; =32
1000005a8: 9400003f    	bl	0x1000006a4 <__Znwm+0x1000006a4>
1000005ac: aa0003f3    	mov	x19, x0
1000005b0: aa0003f4    	mov	x20, x0
1000005b4: f8008e9f    	str	xzr, [x20, #0x8]!
1000005b8: f900081f    	str	xzr, [x0, #0x10]
1000005bc: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005c0: 91008108    	add	x8, x8, #0x20
1000005c4: 91004108    	add	x8, x8, #0x10
1000005c8: f9000008    	str	x8, [x0]
1000005cc: 52800548    	mov	w8, #0x2a               ; =42
1000005d0: b9001808    	str	w8, [x0, #0x18]
1000005d4: 52800028    	mov	w8, #0x1                ; =1
1000005d8: f8280288    	ldadd	x8, x8, [x20]
1000005dc: 92800015    	mov	x21, #-0x1              ; =-1
1000005e0: f8f50288    	ldaddal	x21, x8, [x20]
1000005e4: b4000108    	cbz	x8, 0x100000604 <_main+0x70>
1000005e8: f8f50288    	ldaddal	x21, x8, [x20]
1000005ec: b40001c8    	cbz	x8, 0x100000624 <_main+0x90>
1000005f0: 52800540    	mov	w0, #0x2a               ; =42
1000005f4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000005f8: a9414ff4    	ldp	x20, x19, [sp, #0x10]
1000005fc: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000600: d65f03c0    	ret
100000604: f9400268    	ldr	x8, [x19]
100000608: f9400908    	ldr	x8, [x8, #0x10]
10000060c: aa1303e0    	mov	x0, x19
100000610: d63f0100    	blr	x8
100000614: aa1303e0    	mov	x0, x19
100000618: 9400001a    	bl	0x100000680 <__Znwm+0x100000680>
10000061c: f8f50288    	ldaddal	x21, x8, [x20]
100000620: b5fffe88    	cbnz	x8, 0x1000005f0 <_main+0x5c>
100000624: f9400268    	ldr	x8, [x19]
100000628: f9400908    	ldr	x8, [x8, #0x10]
10000062c: aa1303e0    	mov	x0, x19
100000630: d63f0100    	blr	x8
100000634: aa1303e0    	mov	x0, x19
100000638: 94000012    	bl	0x100000680 <__Znwm+0x100000680>
10000063c: 17ffffed    	b	0x1000005f0 <_main+0x5c>

0000000100000640 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000640: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000644: 91008108    	add	x8, x8, #0x20
100000648: 91004108    	add	x8, x8, #0x10
10000064c: f9000008    	str	x8, [x0]
100000650: 1400000f    	b	0x10000068c <__Znwm+0x10000068c>

0000000100000654 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000654: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000658: 910003fd    	mov	x29, sp
10000065c: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000660: 91008108    	add	x8, x8, #0x20
100000664: 91004108    	add	x8, x8, #0x10
100000668: f9000008    	str	x8, [x0]
10000066c: 94000008    	bl	0x10000068c <__Znwm+0x10000068c>
100000670: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000674: 14000009    	b	0x100000698 <__Znwm+0x100000698>

0000000100000678 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000678: d65f03c0    	ret

000000010000067c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
10000067c: 14000007    	b	0x100000698 <__Znwm+0x100000698>

Disassembly of section __TEXT,__stubs:

0000000100000680 <__stubs>:
100000680: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000684: f9400210    	ldr	x16, [x16]
100000688: d61f0200    	br	x16
10000068c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000690: f9400610    	ldr	x16, [x16, #0x8]
100000694: d61f0200    	br	x16
100000698: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000069c: f9400a10    	ldr	x16, [x16, #0x10]
1000006a0: d61f0200    	br	x16
1000006a4: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006a8: f9400e10    	ldr	x16, [x16, #0x18]
1000006ac: d61f0200    	br	x16
