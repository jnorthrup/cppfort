
/Users/jim/work/cppfort/micro-tests/results/memory/mem104-std-array-at/mem104-std-array-at_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z17test_std_array_atv>:
100000538: d100c3ff    	sub	sp, sp, #0x30
10000053c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000540: 910083fd    	add	x29, sp, #0x20
100000544: 90000008    	adrp	x8, 0x100000000 <___gxx_personality_v0+0x100000000>
100000548: 911c4108    	add	x8, x8, #0x710
10000054c: 3dc00100    	ldr	q0, [x8]
100000550: 910003e0    	mov	x0, sp
100000554: 3d8003e0    	str	q0, [sp]
100000558: b9401108    	ldr	w8, [x8, #0x10]
10000055c: b90013e8    	str	w8, [sp, #0x10]
100000560: d2800041    	mov	x1, #0x2                ; =2
100000564: 94000005    	bl	0x100000578 <__ZNSt3__15arrayIiLm5EE2atB8ne200100Em>
100000568: b9400000    	ldr	w0, [x0]
10000056c: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000570: 9100c3ff    	add	sp, sp, #0x30
100000574: d65f03c0    	ret

0000000100000578 <__ZNSt3__15arrayIiLm5EE2atB8ne200100Em>:
100000578: d100c3ff    	sub	sp, sp, #0x30
10000057c: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000580: 910083fd    	add	x29, sp, #0x20
100000584: f81f83a0    	stur	x0, [x29, #-0x8]
100000588: f9000be1    	str	x1, [sp, #0x10]
10000058c: f85f83a8    	ldur	x8, [x29, #-0x8]
100000590: f90007e8    	str	x8, [sp, #0x8]
100000594: f9400be8    	ldr	x8, [sp, #0x10]
100000598: f1001508    	subs	x8, x8, #0x5
10000059c: 540000a3    	b.lo	0x1000005b0 <__ZNSt3__15arrayIiLm5EE2atB8ne200100Em+0x38>
1000005a0: 14000001    	b	0x1000005a4 <__ZNSt3__15arrayIiLm5EE2atB8ne200100Em+0x2c>
1000005a4: 90000000    	adrp	x0, 0x100000000 <___gxx_personality_v0+0x100000000>
1000005a8: 911c9000    	add	x0, x0, #0x724
1000005ac: 9400000f    	bl	0x1000005e8 <__ZNSt3__120__throw_out_of_rangeB8ne200100EPKc>
1000005b0: f94007e8    	ldr	x8, [sp, #0x8]
1000005b4: f9400be9    	ldr	x9, [sp, #0x10]
1000005b8: 8b090900    	add	x0, x8, x9, lsl #2
1000005bc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000005c0: 9100c3ff    	add	sp, sp, #0x30
1000005c4: d65f03c0    	ret

00000001000005c8 <_main>:
1000005c8: d10083ff    	sub	sp, sp, #0x20
1000005cc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000005d0: 910043fd    	add	x29, sp, #0x10
1000005d4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000005d8: 97ffffd8    	bl	0x100000538 <__Z17test_std_array_atv>
1000005dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005e0: 910083ff    	add	sp, sp, #0x20
1000005e4: d65f03c0    	ret

00000001000005e8 <__ZNSt3__120__throw_out_of_rangeB8ne200100EPKc>:
1000005e8: d100c3ff    	sub	sp, sp, #0x30
1000005ec: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000005f0: 910083fd    	add	x29, sp, #0x20
1000005f4: f81f83a0    	stur	x0, [x29, #-0x8]
1000005f8: d2800200    	mov	x0, #0x10               ; =16
1000005fc: 94000032    	bl	0x1000006c4 <___gxx_personality_v0+0x1000006c4>
100000600: f90003e0    	str	x0, [sp]
100000604: f85f83a1    	ldur	x1, [x29, #-0x8]
100000608: 94000011    	bl	0x10000064c <__ZNSt12out_of_rangeC1B8ne200100EPKc>
10000060c: 14000001    	b	0x100000610 <__ZNSt3__120__throw_out_of_rangeB8ne200100EPKc+0x28>
100000610: f94003e0    	ldr	x0, [sp]
100000614: 90000021    	adrp	x1, 0x100004000 <___gxx_personality_v0+0x100004000>
100000618: f9400821    	ldr	x1, [x1, #0x10]
10000061c: 90000022    	adrp	x2, 0x100004000 <___gxx_personality_v0+0x100004000>
100000620: f9400c42    	ldr	x2, [x2, #0x18]
100000624: 9400002b    	bl	0x1000006d0 <___gxx_personality_v0+0x1000006d0>
100000628: aa0003e8    	mov	x8, x0
10000062c: f94003e0    	ldr	x0, [sp]
100000630: f9000be8    	str	x8, [sp, #0x10]
100000634: aa0103e8    	mov	x8, x1
100000638: b9000fe8    	str	w8, [sp, #0xc]
10000063c: 94000028    	bl	0x1000006dc <___gxx_personality_v0+0x1000006dc>
100000640: 14000001    	b	0x100000644 <__ZNSt3__120__throw_out_of_rangeB8ne200100EPKc+0x5c>
100000644: f9400be0    	ldr	x0, [sp, #0x10]
100000648: 94000028    	bl	0x1000006e8 <___gxx_personality_v0+0x1000006e8>

000000010000064c <__ZNSt12out_of_rangeC1B8ne200100EPKc>:
10000064c: d100c3ff    	sub	sp, sp, #0x30
100000650: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000654: 910083fd    	add	x29, sp, #0x20
100000658: f81f83a0    	stur	x0, [x29, #-0x8]
10000065c: f9000be1    	str	x1, [sp, #0x10]
100000660: f85f83a0    	ldur	x0, [x29, #-0x8]
100000664: f90007e0    	str	x0, [sp, #0x8]
100000668: f9400be1    	ldr	x1, [sp, #0x10]
10000066c: 94000005    	bl	0x100000680 <__ZNSt12out_of_rangeC2B8ne200100EPKc>
100000670: f94007e0    	ldr	x0, [sp, #0x8]
100000674: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000678: 9100c3ff    	add	sp, sp, #0x30
10000067c: d65f03c0    	ret

0000000100000680 <__ZNSt12out_of_rangeC2B8ne200100EPKc>:
100000680: d100c3ff    	sub	sp, sp, #0x30
100000684: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000688: 910083fd    	add	x29, sp, #0x20
10000068c: f81f83a0    	stur	x0, [x29, #-0x8]
100000690: f9000be1    	str	x1, [sp, #0x10]
100000694: f85f83a0    	ldur	x0, [x29, #-0x8]
100000698: f90007e0    	str	x0, [sp, #0x8]
10000069c: f9400be1    	ldr	x1, [sp, #0x10]
1000006a0: 94000015    	bl	0x1000006f4 <___gxx_personality_v0+0x1000006f4>
1000006a4: f94007e0    	ldr	x0, [sp, #0x8]
1000006a8: 90000028    	adrp	x8, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006ac: f9402108    	ldr	x8, [x8, #0x40]
1000006b0: 91004108    	add	x8, x8, #0x10
1000006b4: f9000008    	str	x8, [x0]
1000006b8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000006bc: 9100c3ff    	add	sp, sp, #0x30
1000006c0: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000006c4 <__stubs>:
1000006c4: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006c8: f9400610    	ldr	x16, [x16, #0x8]
1000006cc: d61f0200    	br	x16
1000006d0: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006d4: f9401210    	ldr	x16, [x16, #0x20]
1000006d8: d61f0200    	br	x16
1000006dc: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006e0: f9401610    	ldr	x16, [x16, #0x28]
1000006e4: d61f0200    	br	x16
1000006e8: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006ec: f9401a10    	ldr	x16, [x16, #0x30]
1000006f0: d61f0200    	br	x16
1000006f4: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000006f8: f9401e10    	ldr	x16, [x16, #0x38]
1000006fc: d61f0200    	br	x16
