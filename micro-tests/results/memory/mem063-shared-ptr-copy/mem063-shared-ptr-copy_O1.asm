
/Users/jim/work/cppfort/micro-tests/results/memory/mem063-shared-ptr-copy/mem063-shared-ptr-copy_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z20test_shared_ptr_copyv>:
1000004e8: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004f0: 910043fd    	add	x29, sp, #0x10
1000004f4: 52800400    	mov	w0, #0x20               ; =32
1000004f8: 94000063    	bl	0x100000684 <__Znwm+0x100000684>
1000004fc: aa0003f3    	mov	x19, x0
100000500: f900081f    	str	xzr, [x0, #0x10]
100000504: aa0003e8    	mov	x8, x0
100000508: f8008d1f    	str	xzr, [x8, #0x8]!
10000050c: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
100000510: 91008129    	add	x9, x9, #0x20
100000514: 91004129    	add	x9, x9, #0x10
100000518: f9000009    	str	x9, [x0]
10000051c: 52800549    	mov	w9, #0x2a               ; =42
100000520: b9001809    	str	w9, [x0, #0x18]
100000524: 52800029    	mov	w9, #0x1                ; =1
100000528: f8290109    	ldadd	x9, x9, [x8]
10000052c: 92800014    	mov	x20, #-0x1              ; =-1
100000530: f8f40108    	ldaddal	x20, x8, [x8]
100000534: b50000e8    	cbnz	x8, 0x100000550 <__Z20test_shared_ptr_copyv+0x68>
100000538: f9400268    	ldr	x8, [x19]
10000053c: f9400908    	ldr	x8, [x8, #0x10]
100000540: aa1303e0    	mov	x0, x19
100000544: d63f0100    	blr	x8
100000548: aa1303e0    	mov	x0, x19
10000054c: 94000045    	bl	0x100000660 <__Znwm+0x100000660>
100000550: 91002268    	add	x8, x19, #0x8
100000554: f8f40108    	ldaddal	x20, x8, [x8]
100000558: b50000e8    	cbnz	x8, 0x100000574 <__Z20test_shared_ptr_copyv+0x8c>
10000055c: f9400268    	ldr	x8, [x19]
100000560: f9400908    	ldr	x8, [x8, #0x10]
100000564: aa1303e0    	mov	x0, x19
100000568: d63f0100    	blr	x8
10000056c: aa1303e0    	mov	x0, x19
100000570: 9400003c    	bl	0x100000660 <__Znwm+0x100000660>
100000574: 52800540    	mov	w0, #0x2a               ; =42
100000578: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000057c: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000580: d65f03c0    	ret

0000000100000584 <_main>:
100000584: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000588: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000058c: 910043fd    	add	x29, sp, #0x10
100000590: 52800400    	mov	w0, #0x20               ; =32
100000594: 9400003c    	bl	0x100000684 <__Znwm+0x100000684>
100000598: aa0003f3    	mov	x19, x0
10000059c: f900081f    	str	xzr, [x0, #0x10]
1000005a0: aa0003e8    	mov	x8, x0
1000005a4: f8008d1f    	str	xzr, [x8, #0x8]!
1000005a8: 90000029    	adrp	x9, 0x100004000 <__Znwm+0x100004000>
1000005ac: 91008129    	add	x9, x9, #0x20
1000005b0: 91004129    	add	x9, x9, #0x10
1000005b4: f9000009    	str	x9, [x0]
1000005b8: 52800549    	mov	w9, #0x2a               ; =42
1000005bc: b9001809    	str	w9, [x0, #0x18]
1000005c0: 52800029    	mov	w9, #0x1                ; =1
1000005c4: f8290109    	ldadd	x9, x9, [x8]
1000005c8: 92800014    	mov	x20, #-0x1              ; =-1
1000005cc: f8f40108    	ldaddal	x20, x8, [x8]
1000005d0: b50000e8    	cbnz	x8, 0x1000005ec <_main+0x68>
1000005d4: f9400268    	ldr	x8, [x19]
1000005d8: f9400908    	ldr	x8, [x8, #0x10]
1000005dc: aa1303e0    	mov	x0, x19
1000005e0: d63f0100    	blr	x8
1000005e4: aa1303e0    	mov	x0, x19
1000005e8: 9400001e    	bl	0x100000660 <__Znwm+0x100000660>
1000005ec: 91002268    	add	x8, x19, #0x8
1000005f0: f8f40108    	ldaddal	x20, x8, [x8]
1000005f4: b50000e8    	cbnz	x8, 0x100000610 <_main+0x8c>
1000005f8: f9400268    	ldr	x8, [x19]
1000005fc: f9400908    	ldr	x8, [x8, #0x10]
100000600: aa1303e0    	mov	x0, x19
100000604: d63f0100    	blr	x8
100000608: aa1303e0    	mov	x0, x19
10000060c: 94000015    	bl	0x100000660 <__Znwm+0x100000660>
100000610: 52800540    	mov	w0, #0x2a               ; =42
100000614: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000618: a8c24ff4    	ldp	x20, x19, [sp], #0x20
10000061c: d65f03c0    	ret

0000000100000620 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000620: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000624: 91008108    	add	x8, x8, #0x20
100000628: 91004108    	add	x8, x8, #0x10
10000062c: f9000008    	str	x8, [x0]
100000630: 1400000f    	b	0x10000066c <__Znwm+0x10000066c>

0000000100000634 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000634: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000638: 910003fd    	mov	x29, sp
10000063c: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000640: 91008108    	add	x8, x8, #0x20
100000644: 91004108    	add	x8, x8, #0x10
100000648: f9000008    	str	x8, [x0]
10000064c: 94000008    	bl	0x10000066c <__Znwm+0x10000066c>
100000650: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000654: 14000009    	b	0x100000678 <__Znwm+0x100000678>

0000000100000658 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000658: d65f03c0    	ret

000000010000065c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
10000065c: 14000007    	b	0x100000678 <__Znwm+0x100000678>

Disassembly of section __TEXT,__stubs:

0000000100000660 <__stubs>:
100000660: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000664: f9400210    	ldr	x16, [x16]
100000668: d61f0200    	br	x16
10000066c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000670: f9400610    	ldr	x16, [x16, #0x8]
100000674: d61f0200    	br	x16
100000678: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000067c: f9400a10    	ldr	x16, [x16, #0x10]
100000680: d61f0200    	br	x16
100000684: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000688: f9400e10    	ldr	x16, [x16, #0x18]
10000068c: d61f0200    	br	x16
