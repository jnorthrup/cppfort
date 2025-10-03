
/Users/jim/work/cppfort/micro-tests/results/memory/mem065-weak-ptr/mem065-weak-ptr_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z13test_weak_ptrv>:
1000004e8: a9bd57f6    	stp	x22, x21, [sp, #-0x30]!
1000004ec: a9014ff4    	stp	x20, x19, [sp, #0x10]
1000004f0: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004f4: 910083fd    	add	x29, sp, #0x20
1000004f8: 52800400    	mov	w0, #0x20               ; =32
1000004fc: 9400005a    	bl	0x100000664 <__Znwm+0x100000664>
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
10000052c: 94000045    	bl	0x100000640 <__Znwm+0x100000640>
100000530: b4000220    	cbz	x0, 0x100000574 <__Z13test_weak_ptrv+0x8c>
100000534: b9401a74    	ldr	w20, [x19, #0x18]
100000538: 91002008    	add	x8, x0, #0x8
10000053c: 92800009    	mov	x9, #-0x1               ; =-1
100000540: f8e90108    	ldaddal	x9, x8, [x8]
100000544: b4000288    	cbz	x8, 0x100000594 <__Z13test_weak_ptrv+0xac>
100000548: aa1303e0    	mov	x0, x19
10000054c: 9400003a    	bl	0x100000634 <__Znwm+0x100000634>
100000550: 91002268    	add	x8, x19, #0x8
100000554: 92800009    	mov	x9, #-0x1               ; =-1
100000558: f8e90108    	ldaddal	x9, x8, [x8]
10000055c: b4000348    	cbz	x8, 0x1000005c4 <__Z13test_weak_ptrv+0xdc>
100000560: aa1403e0    	mov	x0, x20
100000564: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000568: a9414ff4    	ldp	x20, x19, [sp, #0x10]
10000056c: a8c357f6    	ldp	x22, x21, [sp], #0x30
100000570: d65f03c0    	ret
100000574: 12800014    	mov	w20, #-0x1              ; =-1
100000578: aa1303e0    	mov	x0, x19
10000057c: 9400002e    	bl	0x100000634 <__Znwm+0x100000634>
100000580: 91002268    	add	x8, x19, #0x8
100000584: 92800009    	mov	x9, #-0x1               ; =-1
100000588: f8e90108    	ldaddal	x9, x8, [x8]
10000058c: b5fffea8    	cbnz	x8, 0x100000560 <__Z13test_weak_ptrv+0x78>
100000590: 1400000d    	b	0x1000005c4 <__Z13test_weak_ptrv+0xdc>
100000594: f9400008    	ldr	x8, [x0]
100000598: f9400908    	ldr	x8, [x8, #0x10]
10000059c: aa0003f5    	mov	x21, x0
1000005a0: d63f0100    	blr	x8
1000005a4: aa1503e0    	mov	x0, x21
1000005a8: 94000023    	bl	0x100000634 <__Znwm+0x100000634>
1000005ac: aa1303e0    	mov	x0, x19
1000005b0: 94000021    	bl	0x100000634 <__Znwm+0x100000634>
1000005b4: 91002268    	add	x8, x19, #0x8
1000005b8: 92800009    	mov	x9, #-0x1               ; =-1
1000005bc: f8e90108    	ldaddal	x9, x8, [x8]
1000005c0: b5fffd08    	cbnz	x8, 0x100000560 <__Z13test_weak_ptrv+0x78>
1000005c4: f9400268    	ldr	x8, [x19]
1000005c8: f9400908    	ldr	x8, [x8, #0x10]
1000005cc: aa1303e0    	mov	x0, x19
1000005d0: d63f0100    	blr	x8
1000005d4: aa1303e0    	mov	x0, x19
1000005d8: 94000017    	bl	0x100000634 <__Znwm+0x100000634>
1000005dc: aa1403e0    	mov	x0, x20
1000005e0: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000005e4: a9414ff4    	ldp	x20, x19, [sp, #0x10]
1000005e8: a8c357f6    	ldp	x22, x21, [sp], #0x30
1000005ec: d65f03c0    	ret

00000001000005f0 <_main>:
1000005f0: 17ffffbe    	b	0x1000004e8 <__Z13test_weak_ptrv>

00000001000005f4 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
1000005f4: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
1000005f8: 9100a108    	add	x8, x8, #0x28
1000005fc: 91004108    	add	x8, x8, #0x10
100000600: f9000008    	str	x8, [x0]
100000604: 14000012    	b	0x10000064c <__Znwm+0x10000064c>

0000000100000608 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000608: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000060c: 910003fd    	mov	x29, sp
100000610: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000614: 9100a108    	add	x8, x8, #0x28
100000618: 91004108    	add	x8, x8, #0x10
10000061c: f9000008    	str	x8, [x0]
100000620: 9400000b    	bl	0x10000064c <__Znwm+0x10000064c>
100000624: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000628: 1400000c    	b	0x100000658 <__Znwm+0x100000658>

000000010000062c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
10000062c: d65f03c0    	ret

0000000100000630 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
100000630: 1400000a    	b	0x100000658 <__Znwm+0x100000658>

Disassembly of section __TEXT,__stubs:

0000000100000634 <__stubs>:
100000634: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000638: f9400210    	ldr	x16, [x16]
10000063c: d61f0200    	br	x16
100000640: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000644: f9400610    	ldr	x16, [x16, #0x8]
100000648: d61f0200    	br	x16
10000064c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000650: f9400a10    	ldr	x16, [x16, #0x10]
100000654: d61f0200    	br	x16
100000658: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000065c: f9400e10    	ldr	x16, [x16, #0x18]
100000660: d61f0200    	br	x16
100000664: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000668: f9401210    	ldr	x16, [x16, #0x20]
10000066c: d61f0200    	br	x16
