
/Users/jim/work/cppfort/micro-tests/results/memory/mem069-enable-shared-from-this/mem069-enable-shared-from-this_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <_main>:
100000538: d100c3ff    	sub	sp, sp, #0x30
10000053c: a9014ff4    	stp	x20, x19, [sp, #0x10]
100000540: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000544: 910083fd    	add	x29, sp, #0x20
100000548: 52800600    	mov	w0, #0x30               ; =48
10000054c: 94000086    	bl	0x100000764 <___gxx_personality_v0+0x100000764>
100000550: aa0003e8    	mov	x8, x0
100000554: f8010d1f    	str	xzr, [x8, #0x10]!
100000558: aa0003e9    	mov	x9, x0
10000055c: f8008d3f    	str	xzr, [x9, #0x8]!
100000560: 9000002a    	adrp	x10, 0x100004000 <___gxx_personality_v0+0x100004000>
100000564: 9101814a    	add	x10, x10, #0x60
100000568: 9100414a    	add	x10, x10, #0x10
10000056c: f900000a    	str	x10, [x0]
100000570: 9100600a    	add	x10, x0, #0x18
100000574: 5280054b    	mov	w11, #0x2a              ; =42
100000578: b900280b    	str	w11, [x0, #0x28]
10000057c: a90003ea    	stp	x10, x0, [sp]
100000580: 5280002b    	mov	w11, #0x1               ; =1
100000584: f82b012c    	ldadd	x11, x12, [x9]
100000588: f82b0108    	ldadd	x11, x8, [x8]
10000058c: a901800a    	stp	x10, x0, [x0, #0x18]
100000590: 92800008    	mov	x8, #-0x1               ; =-1
100000594: f8e80128    	ldaddal	x8, x8, [x9]
100000598: b50000e8    	cbnz	x8, 0x1000005b4 <_main+0x7c>
10000059c: f9400008    	ldr	x8, [x0]
1000005a0: f9400908    	ldr	x8, [x8, #0x10]
1000005a4: aa0003f3    	mov	x19, x0
1000005a8: d63f0100    	blr	x8
1000005ac: aa1303e0    	mov	x0, x19
1000005b0: 94000061    	bl	0x100000734 <___gxx_personality_v0+0x100000734>
1000005b4: f94003e8    	ldr	x8, [sp]
1000005b8: f9400500    	ldr	x0, [x8, #0x8]
1000005bc: b4000400    	cbz	x0, 0x10000063c <_main+0x104>
1000005c0: f9400113    	ldr	x19, [x8]
1000005c4: 9400005f    	bl	0x100000740 <___gxx_personality_v0+0x100000740>
1000005c8: b40003a0    	cbz	x0, 0x10000063c <_main+0x104>
1000005cc: b9401273    	ldr	w19, [x19, #0x10]
1000005d0: 91002008    	add	x8, x0, #0x8
1000005d4: 92800009    	mov	x9, #-0x1               ; =-1
1000005d8: f8e90108    	ldaddal	x9, x8, [x8]
1000005dc: b50000e8    	cbnz	x8, 0x1000005f8 <_main+0xc0>
1000005e0: f9400008    	ldr	x8, [x0]
1000005e4: f9400908    	ldr	x8, [x8, #0x10]
1000005e8: aa0003f4    	mov	x20, x0
1000005ec: d63f0100    	blr	x8
1000005f0: aa1403e0    	mov	x0, x20
1000005f4: 94000050    	bl	0x100000734 <___gxx_personality_v0+0x100000734>
1000005f8: f94007f4    	ldr	x20, [sp, #0x8]
1000005fc: b4000174    	cbz	x20, 0x100000628 <_main+0xf0>
100000600: 91002288    	add	x8, x20, #0x8
100000604: 92800009    	mov	x9, #-0x1               ; =-1
100000608: f8e90108    	ldaddal	x9, x8, [x8]
10000060c: b50000e8    	cbnz	x8, 0x100000628 <_main+0xf0>
100000610: f9400288    	ldr	x8, [x20]
100000614: f9400908    	ldr	x8, [x8, #0x10]
100000618: aa1403e0    	mov	x0, x20
10000061c: d63f0100    	blr	x8
100000620: aa1403e0    	mov	x0, x20
100000624: 94000044    	bl	0x100000734 <___gxx_personality_v0+0x100000734>
100000628: aa1303e0    	mov	x0, x19
10000062c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000630: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000634: 9100c3ff    	add	sp, sp, #0x30
100000638: d65f03c0    	ret
10000063c: 9400001b    	bl	0x1000006a8 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>
100000640: d4200020    	brk	#0x1
100000644: aa0003f3    	mov	x19, x0
100000648: 910003e0    	mov	x0, sp
10000064c: 94000003    	bl	0x100000658 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
100000650: aa1303e0    	mov	x0, x19
100000654: 94000035    	bl	0x100000728 <___gxx_personality_v0+0x100000728>

0000000100000658 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>:
100000658: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
10000065c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000660: 910043fd    	add	x29, sp, #0x10
100000664: f9400413    	ldr	x19, [x0, #0x8]
100000668: b40001b3    	cbz	x19, 0x10000069c <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev+0x44>
10000066c: 91002268    	add	x8, x19, #0x8
100000670: 92800009    	mov	x9, #-0x1               ; =-1
100000674: f8e90108    	ldaddal	x9, x8, [x8]
100000678: b5000128    	cbnz	x8, 0x10000069c <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev+0x44>
10000067c: f9400268    	ldr	x8, [x19]
100000680: f9400908    	ldr	x8, [x8, #0x10]
100000684: aa0003f4    	mov	x20, x0
100000688: aa1303e0    	mov	x0, x19
10000068c: d63f0100    	blr	x8
100000690: aa1303e0    	mov	x0, x19
100000694: 94000028    	bl	0x100000734 <___gxx_personality_v0+0x100000734>
100000698: aa1403e0    	mov	x0, x20
10000069c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000006a0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000006a4: d65f03c0    	ret

00000001000006a8 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>:
1000006a8: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000006ac: 910003fd    	mov	x29, sp
1000006b0: 52800100    	mov	w0, #0x8                ; =8
1000006b4: 9400002f    	bl	0x100000770 <___gxx_personality_v0+0x100000770>
1000006b8: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006bc: f9401508    	ldr	x8, [x8, #0x28]
1000006c0: 91004108    	add	x8, x8, #0x10
1000006c4: f9000008    	str	x8, [x0]
1000006c8: 90000021    	adrp	x1, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006cc: f9401021    	ldr	x1, [x1, #0x20]
1000006d0: 90000022    	adrp	x2, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006d4: f9400042    	ldr	x2, [x2]
1000006d8: 94000029    	bl	0x10000077c <___gxx_personality_v0+0x10000077c>

00000001000006dc <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED1Ev>:
1000006dc: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006e0: 91018108    	add	x8, x8, #0x60
1000006e4: 91004108    	add	x8, x8, #0x10
1000006e8: f9000008    	str	x8, [x0]
1000006ec: 14000018    	b	0x10000074c <___gxx_personality_v0+0x10000074c>

00000001000006f0 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED0Ev>:
1000006f0: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000006f4: 910003fd    	mov	x29, sp
1000006f8: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006fc: 91018108    	add	x8, x8, #0x60
100000700: 91004108    	add	x8, x8, #0x10
100000704: f9000008    	str	x8, [x0]
100000708: 94000011    	bl	0x10000074c <___gxx_personality_v0+0x10000074c>
10000070c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000710: 14000012    	b	0x100000758 <___gxx_personality_v0+0x100000758>

0000000100000714 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
100000714: f9401000    	ldr	x0, [x0, #0x20]
100000718: b4000040    	cbz	x0, 0x100000720 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv+0xc>
10000071c: 14000006    	b	0x100000734 <___gxx_personality_v0+0x100000734>
100000720: d65f03c0    	ret

0000000100000724 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
100000724: 1400000d    	b	0x100000758 <___gxx_personality_v0+0x100000758>

Disassembly of section __TEXT,__stubs:

0000000100000728 <__stubs>:
100000728: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
10000072c: f9402610    	ldr	x16, [x16, #0x48]
100000730: d61f0200    	br	x16
100000734: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000738: f9400610    	ldr	x16, [x16, #0x8]
10000073c: d61f0200    	br	x16
100000740: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000744: f9400a10    	ldr	x16, [x16, #0x10]
100000748: d61f0200    	br	x16
10000074c: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000750: f9400e10    	ldr	x16, [x16, #0x18]
100000754: d61f0200    	br	x16
100000758: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
10000075c: f9402a10    	ldr	x16, [x16, #0x50]
100000760: d61f0200    	br	x16
100000764: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000768: f9402e10    	ldr	x16, [x16, #0x58]
10000076c: d61f0200    	br	x16
100000770: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000774: f9401a10    	ldr	x16, [x16, #0x30]
100000778: d61f0200    	br	x16
10000077c: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000780: f9401e10    	ldr	x16, [x16, #0x38]
100000784: d61f0200    	br	x16
