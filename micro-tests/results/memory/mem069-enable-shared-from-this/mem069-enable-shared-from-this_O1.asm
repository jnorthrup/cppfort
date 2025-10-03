
/Users/jim/work/cppfort/micro-tests/results/memory/mem069-enable-shared-from-this/mem069-enable-shared-from-this_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <_main>:
100000538: d100c3ff    	sub	sp, sp, #0x30
10000053c: a9014ff4    	stp	x20, x19, [sp, #0x10]
100000540: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000544: 910083fd    	add	x29, sp, #0x20
100000548: 52800600    	mov	w0, #0x30               ; =48
10000054c: 94000099    	bl	0x1000007b0 <___gxx_personality_v0+0x1000007b0>
100000550: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
100000554: 91018108    	add	x8, x8, #0x60
100000558: a900fc1f    	stp	xzr, xzr, [x0, #0x8]
10000055c: 91004108    	add	x8, x8, #0x10
100000560: f9000008    	str	x8, [x0]
100000564: f900101f    	str	xzr, [x0, #0x20]
100000568: aa0003e8    	mov	x8, x0
10000056c: f8018d1f    	str	xzr, [x8, #0x18]!
100000570: 52800549    	mov	w9, #0x2a               ; =42
100000574: b9002809    	str	w9, [x0, #0x28]
100000578: a90003e8    	stp	x8, x0, [sp]
10000057c: b400023f    	cbz	xzr, 0x1000005c0 <_main+0x88>
100000580: 52800108    	mov	w8, #0x8                ; =8
100000584: f9400108    	ldr	x8, [x8]
100000588: b100051f    	cmn	x8, #0x1
10000058c: 540003c1    	b.ne	0x100000604 <_main+0xcc>
100000590: 91002008    	add	x8, x0, #0x8
100000594: 52800029    	mov	w9, #0x1                ; =1
100000598: f8290108    	ldadd	x9, x8, [x8]
10000059c: 91004008    	add	x8, x0, #0x10
1000005a0: f8290108    	ldadd	x9, x8, [x8]
1000005a4: 91006008    	add	x8, x0, #0x18
1000005a8: a9018008    	stp	x8, x0, [x0, #0x18]
1000005ac: aa0003f3    	mov	x19, x0
1000005b0: d2800000    	mov	x0, #0x0                ; =0
1000005b4: 94000073    	bl	0x100000780 <___gxx_personality_v0+0x100000780>
1000005b8: aa1303e0    	mov	x0, x19
1000005bc: 14000008    	b	0x1000005dc <_main+0xa4>
1000005c0: 91002008    	add	x8, x0, #0x8
1000005c4: 52800029    	mov	w9, #0x1                ; =1
1000005c8: f8290108    	ldadd	x9, x8, [x8]
1000005cc: 91004008    	add	x8, x0, #0x10
1000005d0: f8290108    	ldadd	x9, x8, [x8]
1000005d4: 91006008    	add	x8, x0, #0x18
1000005d8: a9018008    	stp	x8, x0, [x0, #0x18]
1000005dc: 91002008    	add	x8, x0, #0x8
1000005e0: 92800009    	mov	x9, #-0x1               ; =-1
1000005e4: f8e90108    	ldaddal	x9, x8, [x8]
1000005e8: b50000e8    	cbnz	x8, 0x100000604 <_main+0xcc>
1000005ec: f9400008    	ldr	x8, [x0]
1000005f0: f9400908    	ldr	x8, [x8, #0x10]
1000005f4: aa0003f3    	mov	x19, x0
1000005f8: d63f0100    	blr	x8
1000005fc: aa1303e0    	mov	x0, x19
100000600: 94000060    	bl	0x100000780 <___gxx_personality_v0+0x100000780>
100000604: f94003e8    	ldr	x8, [sp]
100000608: a9400113    	ldp	x19, x0, [x8]
10000060c: b4000040    	cbz	x0, 0x100000614 <_main+0xdc>
100000610: 9400005f    	bl	0x10000078c <___gxx_personality_v0+0x10000078c>
100000614: b40003a0    	cbz	x0, 0x100000688 <_main+0x150>
100000618: b9401273    	ldr	w19, [x19, #0x10]
10000061c: 91002008    	add	x8, x0, #0x8
100000620: 92800009    	mov	x9, #-0x1               ; =-1
100000624: f8e90108    	ldaddal	x9, x8, [x8]
100000628: b50000e8    	cbnz	x8, 0x100000644 <_main+0x10c>
10000062c: f9400008    	ldr	x8, [x0]
100000630: f9400908    	ldr	x8, [x8, #0x10]
100000634: aa0003f4    	mov	x20, x0
100000638: d63f0100    	blr	x8
10000063c: aa1403e0    	mov	x0, x20
100000640: 94000050    	bl	0x100000780 <___gxx_personality_v0+0x100000780>
100000644: f94007f4    	ldr	x20, [sp, #0x8]
100000648: b4000174    	cbz	x20, 0x100000674 <_main+0x13c>
10000064c: 91002288    	add	x8, x20, #0x8
100000650: 92800009    	mov	x9, #-0x1               ; =-1
100000654: f8e90108    	ldaddal	x9, x8, [x8]
100000658: b50000e8    	cbnz	x8, 0x100000674 <_main+0x13c>
10000065c: f9400288    	ldr	x8, [x20]
100000660: f9400908    	ldr	x8, [x8, #0x10]
100000664: aa1403e0    	mov	x0, x20
100000668: d63f0100    	blr	x8
10000066c: aa1403e0    	mov	x0, x20
100000670: 94000044    	bl	0x100000780 <___gxx_personality_v0+0x100000780>
100000674: aa1303e0    	mov	x0, x19
100000678: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000067c: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000680: 9100c3ff    	add	sp, sp, #0x30
100000684: d65f03c0    	ret
100000688: 9400001b    	bl	0x1000006f4 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>
10000068c: d4200020    	brk	#0x1
100000690: aa0003f3    	mov	x19, x0
100000694: 910003e0    	mov	x0, sp
100000698: 94000003    	bl	0x1000006a4 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
10000069c: aa1303e0    	mov	x0, x19
1000006a0: 94000035    	bl	0x100000774 <___gxx_personality_v0+0x100000774>

00000001000006a4 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>:
1000006a4: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
1000006a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000006ac: 910043fd    	add	x29, sp, #0x10
1000006b0: f9400413    	ldr	x19, [x0, #0x8]
1000006b4: b40001b3    	cbz	x19, 0x1000006e8 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev+0x44>
1000006b8: 91002268    	add	x8, x19, #0x8
1000006bc: 92800009    	mov	x9, #-0x1               ; =-1
1000006c0: f8e90108    	ldaddal	x9, x8, [x8]
1000006c4: b5000128    	cbnz	x8, 0x1000006e8 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev+0x44>
1000006c8: f9400268    	ldr	x8, [x19]
1000006cc: f9400908    	ldr	x8, [x8, #0x10]
1000006d0: aa0003f4    	mov	x20, x0
1000006d4: aa1303e0    	mov	x0, x19
1000006d8: d63f0100    	blr	x8
1000006dc: aa1303e0    	mov	x0, x19
1000006e0: 94000028    	bl	0x100000780 <___gxx_personality_v0+0x100000780>
1000006e4: aa1403e0    	mov	x0, x20
1000006e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000006ec: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000006f0: d65f03c0    	ret

00000001000006f4 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>:
1000006f4: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000006f8: 910003fd    	mov	x29, sp
1000006fc: 52800100    	mov	w0, #0x8                ; =8
100000700: 9400002f    	bl	0x1000007bc <___gxx_personality_v0+0x1000007bc>
100000704: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
100000708: f9401508    	ldr	x8, [x8, #0x28]
10000070c: 91004108    	add	x8, x8, #0x10
100000710: f9000008    	str	x8, [x0]
100000714: 90000021    	adrp	x1, 0x100004000 <___gxx_personality_v0+0x100004000>
100000718: f9401021    	ldr	x1, [x1, #0x20]
10000071c: 90000022    	adrp	x2, 0x100004000 <___gxx_personality_v0+0x100004000>
100000720: f9400042    	ldr	x2, [x2]
100000724: 94000029    	bl	0x1000007c8 <___gxx_personality_v0+0x1000007c8>

0000000100000728 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED1Ev>:
100000728: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
10000072c: 91018108    	add	x8, x8, #0x60
100000730: 91004108    	add	x8, x8, #0x10
100000734: f9000008    	str	x8, [x0]
100000738: 14000018    	b	0x100000798 <___gxx_personality_v0+0x100000798>

000000010000073c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED0Ev>:
10000073c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000740: 910003fd    	mov	x29, sp
100000744: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
100000748: 91018108    	add	x8, x8, #0x60
10000074c: 91004108    	add	x8, x8, #0x10
100000750: f9000008    	str	x8, [x0]
100000754: 94000011    	bl	0x100000798 <___gxx_personality_v0+0x100000798>
100000758: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000075c: 14000012    	b	0x1000007a4 <___gxx_personality_v0+0x1000007a4>

0000000100000760 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
100000760: f9401000    	ldr	x0, [x0, #0x20]
100000764: b4000040    	cbz	x0, 0x10000076c <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv+0xc>
100000768: 14000006    	b	0x100000780 <___gxx_personality_v0+0x100000780>
10000076c: d65f03c0    	ret

0000000100000770 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
100000770: 1400000d    	b	0x1000007a4 <___gxx_personality_v0+0x1000007a4>

Disassembly of section __TEXT,__stubs:

0000000100000774 <__stubs>:
100000774: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000778: f9402610    	ldr	x16, [x16, #0x48]
10000077c: d61f0200    	br	x16
100000780: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000784: f9400610    	ldr	x16, [x16, #0x8]
100000788: d61f0200    	br	x16
10000078c: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000790: f9400a10    	ldr	x16, [x16, #0x10]
100000794: d61f0200    	br	x16
100000798: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
10000079c: f9400e10    	ldr	x16, [x16, #0x18]
1000007a0: d61f0200    	br	x16
1000007a4: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000007a8: f9402a10    	ldr	x16, [x16, #0x50]
1000007ac: d61f0200    	br	x16
1000007b0: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000007b4: f9402e10    	ldr	x16, [x16, #0x58]
1000007b8: d61f0200    	br	x16
1000007bc: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000007c0: f9401a10    	ldr	x16, [x16, #0x30]
1000007c4: d61f0200    	br	x16
1000007c8: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000007cc: f9401e10    	ldr	x16, [x16, #0x38]
1000007d0: d61f0200    	br	x16
