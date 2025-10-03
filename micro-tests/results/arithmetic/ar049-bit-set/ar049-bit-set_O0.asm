
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar049-bit-set/ar049-bit-set_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z12test_bit_setii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400bea    	ldr	w10, [sp, #0x8]
100000374: 52800029    	mov	w9, #0x1                ; =1
100000378: 1aca2129    	lsl	w9, w9, w10
10000037c: 2a090100    	orr	w0, w8, w9
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: d10083ff    	sub	sp, sp, #0x20
10000038c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000390: 910043fd    	add	x29, sp, #0x10
100000394: 52800000    	mov	w0, #0x0                ; =0
100000398: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000039c: 52800061    	mov	w1, #0x3                ; =3
1000003a0: 97fffff0    	bl	0x100000360 <__Z12test_bit_setii>
1000003a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a8: 910083ff    	add	sp, sp, #0x20
1000003ac: d65f03c0    	ret
