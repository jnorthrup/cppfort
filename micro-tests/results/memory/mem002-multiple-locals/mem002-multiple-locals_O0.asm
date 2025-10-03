
/Users/jim/work/cppfort/micro-tests/results/memory/mem002-multiple-locals/mem002-multiple-locals_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_multiple_localsv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 52800028    	mov	w8, #0x1                ; =1
100000368: b9000fe8    	str	w8, [sp, #0xc]
10000036c: 52800048    	mov	w8, #0x2                ; =2
100000370: b9000be8    	str	w8, [sp, #0x8]
100000374: 52800068    	mov	w8, #0x3                ; =3
100000378: b90007e8    	str	w8, [sp, #0x4]
10000037c: b9400fe8    	ldr	w8, [sp, #0xc]
100000380: b9400be9    	ldr	w9, [sp, #0x8]
100000384: 0b090108    	add	w8, w8, w9
100000388: b94007e9    	ldr	w9, [sp, #0x4]
10000038c: 0b090100    	add	w0, w8, w9
100000390: 910043ff    	add	sp, sp, #0x10
100000394: d65f03c0    	ret

0000000100000398 <_main>:
100000398: d10083ff    	sub	sp, sp, #0x20
10000039c: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a0: 910043fd    	add	x29, sp, #0x10
1000003a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a8: 97ffffee    	bl	0x100000360 <__Z20test_multiple_localsv>
1000003ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b0: 910083ff    	add	sp, sp, #0x20
1000003b4: d65f03c0    	ret
