
/Users/jim/work/cppfort/micro-tests/results/memory/mem066-weak-ptr-expired/mem066-weak-ptr-expired_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000004e8 <__Z21test_weak_ptr_expiredv>:
1000004e8: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000004ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004f0: 910043fd    	add	x29, sp, #0x10
1000004f4: 52800400    	mov	w0, #0x20               ; =32
1000004f8: 9400005b    	bl	0x100000664 <__Znwm+0x100000664>
1000004fc: aa0003e8    	mov	x8, x0
100000500: f8008d1f    	str	xzr, [x8, #0x8]!
100000504: aa0003e9    	mov	x9, x0
100000508: f8010d3f    	str	xzr, [x9, #0x10]!
10000050c: 9000002a    	adrp	x10, 0x100004000 <__Znwm+0x100004000>
100000510: 9100814a    	add	x10, x10, #0x20
100000514: 9100414a    	add	x10, x10, #0x10
100000518: f900000a    	str	x10, [x0]
10000051c: 5280054a    	mov	w10, #0x2a              ; =42
100000520: b900180a    	str	w10, [x0, #0x18]
100000524: 5280002a    	mov	w10, #0x1               ; =1
100000528: f82a0129    	ldadd	x10, x9, [x9]
10000052c: 92800009    	mov	x9, #-0x1               ; =-1
100000530: f8e90108    	ldaddal	x9, x8, [x8]
100000534: b5000108    	cbnz	x8, 0x100000554 <__Z21test_weak_ptr_expiredv+0x6c>
100000538: f9400008    	ldr	x8, [x0]
10000053c: f9400908    	ldr	x8, [x8, #0x10]
100000540: aa0003f3    	mov	x19, x0
100000544: d63f0100    	blr	x8
100000548: aa1303e0    	mov	x0, x19
10000054c: 9400003d    	bl	0x100000640 <__Znwm+0x100000640>
100000550: aa1303e0    	mov	x0, x19
100000554: f9400408    	ldr	x8, [x0, #0x8]
100000558: b100051f    	cmn	x8, #0x1
10000055c: 1a9f17f3    	cset	w19, eq
100000560: 94000038    	bl	0x100000640 <__Znwm+0x100000640>
100000564: aa1303e0    	mov	x0, x19
100000568: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000056c: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000570: d65f03c0    	ret

0000000100000574 <_main>:
100000574: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000578: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000057c: 910043fd    	add	x29, sp, #0x10
100000580: 52800400    	mov	w0, #0x20               ; =32
100000584: 94000038    	bl	0x100000664 <__Znwm+0x100000664>
100000588: aa0003e8    	mov	x8, x0
10000058c: f8008d1f    	str	xzr, [x8, #0x8]!
100000590: aa0003e9    	mov	x9, x0
100000594: f8010d3f    	str	xzr, [x9, #0x10]!
100000598: 9000002a    	adrp	x10, 0x100004000 <__Znwm+0x100004000>
10000059c: 9100814a    	add	x10, x10, #0x20
1000005a0: 9100414a    	add	x10, x10, #0x10
1000005a4: f900000a    	str	x10, [x0]
1000005a8: 5280054a    	mov	w10, #0x2a              ; =42
1000005ac: b900180a    	str	w10, [x0, #0x18]
1000005b0: 5280002a    	mov	w10, #0x1               ; =1
1000005b4: f82a0129    	ldadd	x10, x9, [x9]
1000005b8: 92800009    	mov	x9, #-0x1               ; =-1
1000005bc: f8e90108    	ldaddal	x9, x8, [x8]
1000005c0: b5000108    	cbnz	x8, 0x1000005e0 <_main+0x6c>
1000005c4: f9400008    	ldr	x8, [x0]
1000005c8: f9400908    	ldr	x8, [x8, #0x10]
1000005cc: aa0003f3    	mov	x19, x0
1000005d0: d63f0100    	blr	x8
1000005d4: aa1303e0    	mov	x0, x19
1000005d8: 9400001a    	bl	0x100000640 <__Znwm+0x100000640>
1000005dc: aa1303e0    	mov	x0, x19
1000005e0: f9400408    	ldr	x8, [x0, #0x8]
1000005e4: b100051f    	cmn	x8, #0x1
1000005e8: 1a9f17f3    	cset	w19, eq
1000005ec: 94000015    	bl	0x100000640 <__Znwm+0x100000640>
1000005f0: aa1303e0    	mov	x0, x19
1000005f4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005f8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005fc: d65f03c0    	ret

0000000100000600 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED1Ev>:
100000600: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000604: 91008108    	add	x8, x8, #0x20
100000608: 91004108    	add	x8, x8, #0x10
10000060c: f9000008    	str	x8, [x0]
100000610: 1400000f    	b	0x10000064c <__Znwm+0x10000064c>

0000000100000614 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEED0Ev>:
100000614: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000618: 910003fd    	mov	x29, sp
10000061c: 90000028    	adrp	x8, 0x100004000 <__Znwm+0x100004000>
100000620: 91008108    	add	x8, x8, #0x20
100000624: 91004108    	add	x8, x8, #0x10
100000628: f9000008    	str	x8, [x0]
10000062c: 94000008    	bl	0x10000064c <__Znwm+0x10000064c>
100000630: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000634: 14000009    	b	0x100000658 <__Znwm+0x100000658>

0000000100000638 <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE16__on_zero_sharedEv>:
100000638: d65f03c0    	ret

000000010000063c <__ZNSt3__120__shared_ptr_emplaceIiNS_9allocatorIiEEE21__on_zero_shared_weakEv>:
10000063c: 14000007    	b	0x100000658 <__Znwm+0x100000658>

Disassembly of section __TEXT,__stubs:

0000000100000640 <__stubs>:
100000640: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000644: f9400210    	ldr	x16, [x16]
100000648: d61f0200    	br	x16
10000064c: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000650: f9400610    	ldr	x16, [x16, #0x8]
100000654: d61f0200    	br	x16
100000658: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
10000065c: f9400a10    	ldr	x16, [x16, #0x10]
100000660: d61f0200    	br	x16
100000664: 90000030    	adrp	x16, 0x100004000 <__Znwm+0x100004000>
100000668: f9400e10    	ldr	x16, [x16, #0x18]
10000066c: d61f0200    	br	x16
