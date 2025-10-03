
/Users/jim/work/cppfort/micro-tests/results/memory/mem064-shared-ptr-use-count/mem064-shared-ptr-use-count_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z25test_shared_ptr_use_countv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 94000068    	bl	0x10000069c <__Znwm+0x10000069c>
100000500: aa0003f3    	mov	x19, x0
100000504: aa0003e8    	mov	x8, x0
100000508: f8008d1f    	str	xzr, [x8, #0x8]!
10000050c: f900081f    	str	xzr, [x0, #0x10]
100000510: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000514: 91008129    	add	x9, x9, #0x20
100000518: 91004129    	add	x9, x9, #0x10
10000051c: f9000009    	str	x9, [x0]
100000520: 52800549    	mov	w9, #0x2a               ; =42
100000524: b9001809    	str	w9, [x0, #0x18]
100000528: 52800029    	mov	w9, #0x1                ; =1
10000052c: f8290109    	ldadd	x9, x9, [x8]
100000530: f9400114    	ldr	x20, [x8]
100000534: 92800015    	mov	x21, #-0x1              ; =-1
100000538: f8f50108    	ldaddal	x21, x8, [x8]
10000053c: b50000e8    	cbnz	x8, 0x100000558 <__Z25test_shared_ptr_use_countv+0x70>
100000540: f9400268    	ldr	x8, [x19]
100000544: f9400908    	ldr	x8, [x8, #0x10]
100000548: aa1303e0    	mov	x0, x19
10000054c: d63f0100    	blr	x8
100000550: aa1303e0    	mov	x0, x19
100000554: 94000049    	bl	0x100000678 <__Znwm+0x100000678>
100000558: 91002268    	add	x8, x19, #0x8
10000055c: f8f50108    	ldaddal	x21, x8, [x8]
100000560: b50000e8    	cbnz	x8, 0x10000057c <__Z25test_shared_ptr_use_countv+0x94>
100000564: f9400268    	ldr	x8, [x19]
100000568: f9400908    	ldr	x8, [x8, #0x10]
10000056c: aa1303e0    	mov	x0, x19
100000570: d63f0100    	blr	x8
100000574: aa1303e0    	mov	x0, x19
100000578: 94000040    	bl	0x100000678 <__Znwm+0x100000678>
10000057c: 11000680    	add	w0, w20, #0x1
100000580: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000584: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000588: a8c357f6    	ldp	x22, x21, [sp], #0x30
10000058c: d65f03c0    	ret

0000000100000590 <_main>:
100000590: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
100000594: a9014ff4    	stp	x20, x19, [sp, #0x10]
100000598: a9027bfd    	stp	x29, x30, [sp, #0x20]
10000059c: 910083fd    	add	x29, sp, #0x20
1000005a0: 52800400    	mov	w0, #0x20               ; =32
1000005a4: 9400003e    	bl	0x10000069c <__Znwm+0x10000069c>
1000005a8: aa0003f3    	mov	x19, x0
1000005ac: aa0003e8    	mov	x8, x0
1000005b0: f8008d1f    	str	xzr, [x8, #0x8]!
1000005b4: f900081f    	str	xzr, [x0, #0x10]
1000005b8: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
1000005bc: 91008129    	add	x9, x9, #0x20
1000005c0: 91004129    	add	x9, x9, #0x10
1000005c4: f9000009    	str	x9, [x0]
1000005c8: 52800549    	mov	w9, #0x2a               ; =42
1000005cc: b9001809    	str	w9, [x0, #0x18]
1000005d0: 52800029    	mov	w9, #0x1                ; =1
1000005d4: f8290109    	ldadd	x9, x9, [x8]
1000005d8: f9400114    	ldr	x20, [x8]
1000005dc: 92800015    	mov	x21, #-0x1              ; =-1
1000005e0: f8f50108    	ldaddal	x21, x8, [x8]
1000005e4: b50000e8    	cbnz	x8, 0x100000600 <_main+0x70>
1000005e8: f9400268    	ldr	x8, [x19]
1000005ec: f9400908    	ldr	x8, [x8, #0x10]
1000005f0: aa1303e0    	mov	x0, x19
1000005f4: d63f0100    	blr	x8
1000005f8: aa1303e0    	mov	x0, x19
1000005fc: 9400001f    	bl	0x100000678 <__Znwm+0x100000678>
100000600: 91002268    	add	x8, x19, #0x8
100000604: f8f50108    	ldaddal	x21, x8, [x8]
100000608: b50000e8    	cbnz	x8, 0x100000624 <_main+0x94>
10000060c: f9400268    	ldr	x8, [x19]
100000610: f9400908    	ldr	x8, [x8, #0x10]
100000614: aa1303e0    	mov	x0, x19
100000618: d63f0100    	blr	x8
10000061c: aa1303e0    	mov	x0, x19
100000620: 94000016    	bl	0x100000678 <__Znwm+0x100000678>
100000624: 11000680    	add	w0, w20, #0x1
100000628: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000062c: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000630: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000634: d65f03c0    	ret

0000000100000638 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000638: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
10000063c: 91008108    	add	x8, x8, #0x20
100000640: 91004108    	add	x8, x8, #0x10
100000644: f9000008    	str	x8, [x0]
100000648: 1400000f    	b	0x100000684 <__Znwm+0x100000684>

000000010000064c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
10000064c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000650: 910003fd    	mov	x29, sp
100000654: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000658: 91008108    	add	x8, x8, #0x20
10000065c: 91004108    	add	x8, x8, #0x10
100000660: f9000008    	str	x8, [x0]
100000664: 94000008    	bl	0x100000684 <__Znwm+0x100000684>
100000668: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000066c: 14000009    	b	0x100000690 <__Znwm+0x100000690>

0000000100000670 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000670: d65f03c0    	ret

0000000100000674 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000674: 14000007    	b	0x100000690 <__Znwm+0x100000690>

Disassembly of section __TEXT,__stubs:

0000000100000678 <__stubs>:
100000678: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000067c: f9400210    	ldr	x16, [x16]
100000680: d61f0200    	br	x16
100000684: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000688: f9400610    	ldr	x16, [x16, #0x8]
10000068c: d61f0200    	br	x16
100000690: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000694: f9400a10    	ldr	x16, [x16, #0x10]
100000698: d61f0200    	br	x16
10000069c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
1000006a0: f9400e10    	ldr	x16, [x16, #0x18]
1000006a4: d61f0200    	br	x16
