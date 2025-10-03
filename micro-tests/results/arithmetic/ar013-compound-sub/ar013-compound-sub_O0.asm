
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar013-compound-sub/ar013-compound-sub_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_compound_subii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400be9    	ldr	w9, [sp, #0x8]
100000370: b9400fe8    	ldr	w8, [sp, #0xc]
100000374: 6b090108    	subs	w8, w8, w9
100000378: b9000fe8    	str	w8, [sp, #0xc]
10000037c: b9400fe0    	ldr	w0, [sp, #0xc]
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: d10083ff    	sub	sp, sp, #0x20
10000038c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000390: 910043fd    	add	x29, sp, #0x10
100000394: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000398: 52800140    	mov	w0, #0xa                ; =10
10000039c: 52800061    	mov	w1, #0x3                ; =3
1000003a0: 97fffff0    	bl	0x100000360 <__Z17test_compound_subii>
1000003a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a8: 910083ff    	add	sp, sp, #0x20
1000003ac: d65f03c0    	ret
