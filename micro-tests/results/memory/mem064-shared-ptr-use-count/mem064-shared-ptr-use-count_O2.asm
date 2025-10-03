
/Users/jim/work/cppfort/micro-tests/results/memory/mem064-shared-ptr-use-count/mem064-shared-ptr-use-count_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z25test_shared_ptr_use_countv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 9400006c    	bl	0x1000006ac <__Znwm+0x1000006ac>
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
100000570: 94000046    	bl	0x100000688 <__Znwm+0x100000688>
100000574: f8f602a8    	ldaddal	x22, x8, [x21]
100000578: b5fffe88    	cbnz	x8, 0x100000548 <__Z25test_shared_ptr_use_countv+0x60>
10000057c: f9400268    	ldr	x8, [x19]
100000580: f9400908    	ldr	x8, [x8, #0x10]
100000584: aa1303e0    	mov	x0, x19
100000588: d63f0100    	blr	x8
10000058c: aa1303e0    	mov	x0, x19
100000590: 9400003e    	bl	0x100000688 <__Znwm+0x100000688>
100000594: 17ffffed    	b	0x100000548 <__Z25test_shared_ptr_use_countv+0x60>

0000000100000598 <_main>:
100000598: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
10000059c: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000005a0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005a4: 910083fd    	add	x29, sp, #0x20
1000005a8: 52800400    	mov	w0, #0x20               ; =32
1000005ac: 94000040    	bl	0x1000006ac <__Znwm+0x1000006ac>
1000005b0: aa0003f3    	mov	x19, x0
1000005b4: aa0003f5    	mov	x21, x0
1000005b8: f8008ebf    	str	xzr, [x21, #0x8]!
1000005bc: f900081f    	str	xzr, [x0, #0x10]
1000005c0: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005c4: 91008108    	add	x8, x8, #0x20
1000005c8: 91004108    	add	x8, x8, #0x10
1000005cc: f9000008    	str	x8, [x0]
1000005d0: 52800548    	mov	w8, #0x2a               ; =42
1000005d4: b9001808    	str	w8, [x0, #0x18]
1000005d8: 52800028    	mov	w8, #0x1                ; =1
1000005dc: f82802a8    	ldadd	x8, x8, [x21]
1000005e0: f94002b4    	ldr	x20, [x21]
1000005e4: 92800016    	mov	x22, #-0x1              ; =-1
1000005e8: f8f602a8    	ldaddal	x22, x8, [x21]
1000005ec: b4000108    	cbz	x8, 0x10000060c <_main+0x74>
1000005f0: f8f602a8    	ldaddal	x22, x8, [x21]
1000005f4: b40001c8    	cbz	x8, 0x10000062c <_main+0x94>
1000005f8: 11000680    	add	w0, w20, #0x1
1000005fc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000600: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000604: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000608: d65f03c0    	ret
10000060c: f9400268    	ldr	x8, [x19]
100000610: f9400908    	ldr	x8, [x8, #0x10]
100000614: aa1303e0    	mov	x0, x19
100000618: d63f0100    	blr	x8
10000061c: aa1303e0    	mov	x0, x19
100000620: 9400001a    	bl	0x100000688 <__Znwm+0x100000688>
100000624: f8f602a8    	ldaddal	x22, x8, [x21]
100000628: b5fffe88    	cbnz	x8, 0x1000005f8 <_main+0x60>
10000062c: f9400268    	ldr	x8, [x19]
100000630: f9400908    	ldr	x8, [x8, #0x10]
100000634: aa1303e0    	mov	x0, x19
100000638: d63f0100    	blr	x8
10000063c: aa1303e0    	mov	x0, x19
100000640: 94000012    	bl	0x100000688 <__Znwm+0x100000688>
100000644: 17ffffed    	b	0x1000005f8 <_main+0x60>

0000000100000648 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000648: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
10000064c: 91008108    	add	x8, x8, #0x20
100000650: 91004108    	add	x8, x8, #0x10
100000654: f9000008    	str	x8, [x0]
100000658: 1400000f    	b	0x100000694 <__Znwm+0x100000694>

000000010000065c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
10000065c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000660: 910003fd    	mov	x29, sp
100000664: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000668: 91008108    	add	x8, x8, #0x20
10000066c: 91004108    	add	x8, x8, #0x10
100000670: f9000008    	str	x8, [x0]
100000674: 94000008    	bl	0x100000694 <__Znwm+0x100000694>
100000678: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000067c: 14000009    	b	0x1000006a0 <__Znwm+0x1000006a0>

0000000100000680 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000680: d65f03c0    	ret

0000000100000684 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000684: 14000007    	b	0x1000006a0 <__Znwm+0x1000006a0>

Disassembly of section __TEXT,__stubs:

0000000100000688 <__stubs>:
100000688: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000068c: f9400210    	ldr	x16, [x16]
100000690: d61f0200    	br	x16
100000694: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000698: f9400610    	ldr	x16, [x16, #0x8]
10000069c: d61f0200    	br	x16
1000006a0: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006a4: f9400a10    	ldr	x16, [x16, #0x10]
1000006a8: d61f0200    	br	x16
1000006ac: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006b0: f9400e10    	ldr	x16, [x16, #0x18]
1000006b4: d61f0200    	br	x16
