
/Users/jim/work/cppfort/micro-tests/results/memory/mem069-enable-shared-from-this/mem069-enable-shared-from-this_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <_main>:
100000538: d100c3ff    	sub	sp, sp, #0x30
10000053c: a9014ff4    	stp	x20, x19, [sp, #0x10]
100000540: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000544: 910083fd    	add	x29, sp, #0x20
100000548: 52800600    	mov	w0, #0x30               ; =48
10000054c: 9400008e    	bl	0x100000784 <___gxx_personality_v0+0x100000784>
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
1000005b0: 94000069    	bl	0x100000754 <___gxx_personality_v0+0x100000754>
1000005b4: f94003e8    	ldr	x8, [sp]
1000005b8: f9400500    	ldr	x0, [x8, #0x8]
1000005bc: b40004a0    	cbz	x0, 0x100000650 <_main+0x118>
1000005c0: f9400113    	ldr	x19, [x8]
1000005c4: 94000067    	bl	0x100000760 <___gxx_personality_v0+0x100000760>
1000005c8: b4000440    	cbz	x0, 0x100000650 <_main+0x118>
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
1000005f4: 94000058    	bl	0x100000754 <___gxx_personality_v0+0x100000754>
1000005f8: f94007f4    	ldr	x20, [sp, #0x8]
1000005fc: b40000b4    	cbz	x20, 0x100000610 <_main+0xd8>
100000600: 91002288    	add	x8, x20, #0x8
100000604: 92800009    	mov	x9, #-0x1               ; =-1
100000608: f8e90108    	ldaddal	x9, x8, [x8]
10000060c: b40000c8    	cbz	x8, 0x100000624 <_main+0xec>
100000610: aa1303e0    	mov	x0, x19
100000614: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000618: a9414ff4    	ldp	x20, x19, [sp, #0x10]
10000061c: 9100c3ff    	add	sp, sp, #0x30
100000620: d65f03c0    	ret
100000624: f9400288    	ldr	x8, [x20]
100000628: f9400908    	ldr	x8, [x8, #0x10]
10000062c: aa1403e0    	mov	x0, x20
100000630: d63f0100    	blr	x8
100000634: aa1403e0    	mov	x0, x20
100000638: 94000047    	bl	0x100000754 <___gxx_personality_v0+0x100000754>
10000063c: aa1303e0    	mov	x0, x19
100000640: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000644: a9414ff4    	ldp	x20, x19, [sp, #0x10]
100000648: 9100c3ff    	add	sp, sp, #0x30
10000064c: d65f03c0    	ret
100000650: 9400001e    	bl	0x1000006c8 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>
100000654: d4200020    	brk	#0x1
100000658: aa0003f3    	mov	x19, x0
10000065c: 910003e0    	mov	x0, sp
100000660: 94000003    	bl	0x10000066c <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>
100000664: aa1303e0    	mov	x0, x19
100000668: 94000038    	bl	0x100000748 <___gxx_personality_v0+0x100000748>

000000010000066c <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev>:
10000066c: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000670: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000674: 910043fd    	add	x29, sp, #0x10
100000678: f9400413    	ldr	x19, [x0, #0x8]
10000067c: b40000b3    	cbz	x19, 0x100000690 <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev+0x24>
100000680: 91002268    	add	x8, x19, #0x8
100000684: 92800009    	mov	x9, #-0x1               ; =-1
100000688: f8e90108    	ldaddal	x9, x8, [x8]
10000068c: b4000088    	cbz	x8, 0x10000069c <__ZNSt3__110shared_ptrI4NodeED1B8ne200100Ev+0x30>
100000690: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000694: a8c24ff4    	ldp	x20, x19, [sp], #0x20
100000698: d65f03c0    	ret
10000069c: f9400268    	ldr	x8, [x19]
1000006a0: f9400908    	ldr	x8, [x8, #0x10]
1000006a4: aa0003f4    	mov	x20, x0
1000006a8: aa1303e0    	mov	x0, x19
1000006ac: d63f0100    	blr	x8
1000006b0: aa1303e0    	mov	x0, x19
1000006b4: 94000028    	bl	0x100000754 <___gxx_personality_v0+0x100000754>
1000006b8: aa1403e0    	mov	x0, x20
1000006bc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000006c0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000006c4: d65f03c0    	ret

00000001000006c8 <__ZNSt3__120__throw_bad_weak_ptrB8ne200100Ev>:
1000006c8: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
1000006cc: 910003fd    	mov	x29, sp
1000006d0: 52800100    	mov	w0, #0x8                ; =8
1000006d4: 9400002f    	bl	0x100000790 <___gxx_personality_v0+0x100000790>
1000006d8: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006dc: f9401508    	ldr	x8, [x8, #0x28]
1000006e0: 91004108    	add	x8, x8, #0x10
1000006e4: f9000008    	str	x8, [x0]
1000006e8: 90000021    	adrp	x1, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006ec: f9401021    	ldr	x1, [x1, #0x20]
1000006f0: 90000022    	adrp	x2, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006f4: f9400042    	ldr	x2, [x2]
1000006f8: 94000029    	bl	0x10000079c <___gxx_personality_v0+0x10000079c>

00000001000006fc <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED1Ev>:
1000006fc: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
100000700: 91018108    	add	x8, x8, #0x60
100000704: 91004108    	add	x8, x8, #0x10
100000708: f9000008    	str	x8, [x0]
10000070c: 14000018    	b	0x10000076c <___gxx_personality_v0+0x10000076c>

0000000100000710 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEED0Ev>:
100000710: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000714: 910003fd    	mov	x29, sp
100000718: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
10000071c: 91018108    	add	x8, x8, #0x60
100000720: 91004108    	add	x8, x8, #0x10
100000724: f9000008    	str	x8, [x0]
100000728: 94000011    	bl	0x10000076c <___gxx_personality_v0+0x10000076c>
10000072c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000730: 14000012    	b	0x100000778 <___gxx_personality_v0+0x100000778>

0000000100000734 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv>:
100000734: f9401000    	ldr	x0, [x0, #0x20]
100000738: b4000040    	cbz	x0, 0x100000740 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE16__on_zero_sharedEv+0xc>
10000073c: 14000006    	b	0x100000754 <___gxx_personality_v0+0x100000754>
100000740: d65f03c0    	ret

0000000100000744 <__ZNSt3__120__shared_ptr_emplaceI4NodeNS_9allocatorIS1_EEE21__on_zero_shared_weakEv>:
100000744: 1400000d    	b	0x100000778 <___gxx_personality_v0+0x100000778>

Disassembly of section __TEXT,__stubs:

0000000100000748 <__stubs>:
100000748: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
10000074c: f9402610    	ldr	x16, [x16, #0x48]
100000750: d61f0200    	br	x16
100000754: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000758: f9400610    	ldr	x16, [x16, #0x8]
10000075c: d61f0200    	br	x16
100000760: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000764: f9400a10    	ldr	x16, [x16, #0x10]
100000768: d61f0200    	br	x16
10000076c: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000770: f9400e10    	ldr	x16, [x16, #0x18]
100000774: d61f0200    	br	x16
100000778: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
10000077c: f9402a10    	ldr	x16, [x16, #0x50]
100000780: d61f0200    	br	x16
100000784: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000788: f9402e10    	ldr	x16, [x16, #0x58]
10000078c: d61f0200    	br	x16
100000790: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
100000794: f9401a10    	ldr	x16, [x16, #0x30]
100000798: d61f0200    	br	x16
10000079c: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000007a0: f9401e10    	ldr	x16, [x16, #0x38]
1000007a4: d61f0200    	br	x16
