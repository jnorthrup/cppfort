
/Users/jim/work/cppfort/micro-tests/results/memory/mem064-shared-ptr-use-count/mem064-shared-ptr-use-count_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z25test_shared_ptr_use_countv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 94000074    	bl	0x1000006cc <__Znwm+0x1000006cc>
100000500: aa0003f3    	mov	x19, x0
100000504: aa0003f5    	mov	x21, x0
100000508: f8008ebf    	str	xzr, [x21, #0x8]!
10000050c: f900081f    	str	xzr, [x0, #0x10]
100000510: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000514: 91008108    	add	x8, x8, #0x20
100000518: 91004108    	add	x8, x8, #0x10
10000051c: f9000008    	str	x8, [x0]
100000520: 52800548    	mov	w8, #0x2a               ; =42
100000524: b9001808    	str	w8, [x0, #0x18]
100000528: 52800028    	mov	w8, #0x1                ; =1
10000052c: f82802a8    	ldadd	x8, x8, [x21]
100000530: f94002b4    	ldr	x20, [x21]
100000534: 92800016    	mov	x22, #-0x1              ; =-1
100000538: f8f602a8    	ldaddal	x22, x8, [x21]
10000053c: b4000108    	cbz	x8, 0x10000055c <__Z25test_shared_ptr_use_countv+0x74>
100000540: f8f602a8    	ldaddal	x22, x8, [x21]
100000544: b40001c8    	cbz	x8, 0x10000057c <__Z25test_shared_ptr_use_countv+0x94>
100000548: 11000680    	add	w0, w20, #0x1
10000054c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000550: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000554: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000558: d65f03c0    	ret
10000055c: f9400268    	ldr	x8, [x19]
100000560: f9400908    	ldr	x8, [x8, #0x10]
100000564: aa1303e0    	mov	x0, x19
100000568: d63f0100    	blr	x8
10000056c: aa1303e0    	mov	x0, x19
100000570: 9400004e    	bl	0x1000006a8 <__Znwm+0x1000006a8>
100000574: f8f602a8    	ldaddal	x22, x8, [x21]
100000578: b5fffe88    	cbnz	x8, 0x100000548 <__Z25test_shared_ptr_use_countv+0x60>
10000057c: f9400268    	ldr	x8, [x19]
100000580: f9400908    	ldr	x8, [x8, #0x10]
100000584: aa1303e0    	mov	x0, x19
100000588: d63f0100    	blr	x8
10000058c: aa1303e0    	mov	x0, x19
100000590: 94000046    	bl	0x1000006a8 <__Znwm+0x1000006a8>
100000594: 11000680    	add	w0, w20, #0x1
100000598: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000059c: a9414ff4    	ldp	x20, x19, [sp, #0x10]
1000005a0: a8c357f6    	ldp	x22, x21, [sp], #0x30
1000005a4: d65f03c0    	ret

00000001000005a8 <_main>:
1000005a8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000005ac: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000005b0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005b4: 910083fd    	add	x29, sp, #0x20
1000005b8: 52800400    	mov	w0, #0x20               ; =32
1000005bc: 94000044    	bl	0x1000006cc <__Znwm+0x1000006cc>
1000005c0: aa0003f3    	mov	x19, x0
1000005c4: aa0003f5    	mov	x21, x0
1000005c8: f8008ebf    	str	xzr, [x21, #0x8]!
1000005cc: f900081f    	str	xzr, [x0, #0x10]
1000005d0: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005d4: 91008108    	add	x8, x8, #0x20
1000005d8: 91004108    	add	x8, x8, #0x10
1000005dc: f9000008    	str	x8, [x0]
1000005e0: 52800548    	mov	w8, #0x2a               ; =42
1000005e4: b9001808    	str	w8, [x0, #0x18]
1000005e8: 52800028    	mov	w8, #0x1                ; =1
1000005ec: f82802a8    	ldadd	x8, x8, [x21]
1000005f0: f94002b4    	ldr	x20, [x21]
1000005f4: 92800016    	mov	x22, #-0x1              ; =-1
1000005f8: f8f602a8    	ldaddal	x22, x8, [x21]
1000005fc: b4000108    	cbz	x8, 0x10000061c <_main+0x74>
100000600: f8f602a8    	ldaddal	x22, x8, [x21]
100000604: b40001c8    	cbz	x8, 0x10000063c <_main+0x94>
100000608: 11000680    	add	w0, w20, #0x1
10000060c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000610: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000614: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000618: d65f03c0    	ret
10000061c: f9400268    	ldr	x8, [x19]
100000620: f9400908    	ldr	x8, [x8, #0x10]
100000624: aa1303e0    	mov	x0, x19
100000628: d63f0100    	blr	x8
10000062c: aa1303e0    	mov	x0, x19
100000630: 9400001e    	bl	0x1000006a8 <__Znwm+0x1000006a8>
100000634: f8f602a8    	ldaddal	x22, x8, [x21]
100000638: b5fffe88    	cbnz	x8, 0x100000608 <_main+0x60>
10000063c: f9400268    	ldr	x8, [x19]
100000640: f9400908    	ldr	x8, [x8, #0x10]
100000644: aa1303e0    	mov	x0, x19
100000648: d63f0100    	blr	x8
10000064c: aa1303e0    	mov	x0, x19
100000650: 94000016    	bl	0x1000006a8 <__Znwm+0x1000006a8>
100000654: 11000680    	add	w0, w20, #0x1
100000658: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000065c: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000660: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000664: d65f03c0    	ret

0000000100000668 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000668: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
10000066c: 91008108    	add	x8, x8, #0x20
100000670: 91004108    	add	x8, x8, #0x10
100000674: f9000008    	str	x8, [x0]
100000678: 1400000f    	b	0x1000006b4 <__Znwm+0x1000006b4>

000000010000067c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
10000067c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000680: 910003fd    	mov	x29, sp
100000684: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000688: 91008108    	add	x8, x8, #0x20
10000068c: 91004108    	add	x8, x8, #0x10
100000690: f9000008    	str	x8, [x0]
100000694: 94000008    	bl	0x1000006b4 <__Znwm+0x1000006b4>
100000698: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000069c: 14000009    	b	0x1000006c0 <__Znwm+0x1000006c0>

00000001000006a0 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
1000006a0: d65f03c0    	ret

00000001000006a4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
1000006a4: 14000007    	b	0x1000006c0 <__Znwm+0x1000006c0>

Disassembly of section __TEXT,__stubs:

00000001000006a8 <__stubs>:
1000006a8: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006ac: f9400210    	ldr	x16, [x16]
1000006b0: d61f0200    	br	x16
1000006b4: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006b8: f9400610    	ldr	x16, [x16, #0x8]
1000006bc: d61f0200    	br	x16
1000006c0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006c4: f9400a10    	ldr	x16, [x16, #0x10]
1000006c8: d61f0200    	br	x16
1000006cc: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006d0: f9400e10    	ldr	x16, [x16, #0x18]
1000006d4: d61f0200    	br	x16
