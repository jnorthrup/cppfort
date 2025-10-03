
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar052-bit-test/ar052-bit-test_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13test_bit_testii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400bea    	ldr	w10, [sp, #0x8]
100000374: 52800029    	mov	w9, #0x1                ; =1
100000378: 1aca2129    	lsl	w9, w9, w10
10000037c: 6a090108    	ands	w8, w8, w9
100000380: 1a9f07e0    	cset	w0, ne
100000384: 910043ff    	add	sp, sp, #0x10
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: d10083ff    	sub	sp, sp, #0x20
100000390: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000394: 910043fd    	add	x29, sp, #0x10
100000398: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000039c: 52800100    	mov	w0, #0x8                ; =8
1000003a0: 52800061    	mov	w1, #0x3                ; =3
1000003a4: 97ffffef    	bl	0x100000360 <__Z13test_bit_testii>
1000003a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ac: 910083ff    	add	sp, sp, #0x20
1000003b0: d65f03c0    	ret
