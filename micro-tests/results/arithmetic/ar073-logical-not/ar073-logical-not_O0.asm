
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar073-logical-not/ar073-logical-not_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_logical_notb>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39003fe0    	strb	w0, [sp, #0xf]
100000368: 39403fe8    	ldrb	w8, [sp, #0xf]
10000036c: 52000108    	eor	w8, w8, #0x1
100000370: 12000100    	and	w0, w8, #0x1
100000374: 910043ff    	add	sp, sp, #0x10
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: d10083ff    	sub	sp, sp, #0x20
100000380: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000384: 910043fd    	add	x29, sp, #0x10
100000388: 52800008    	mov	w8, #0x0                ; =0
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 12000100    	and	w0, w8, #0x1
100000394: 97fffff3    	bl	0x100000360 <__Z16test_logical_notb>
100000398: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000039c: 910083ff    	add	sp, sp, #0x20
1000003a0: d65f03c0    	ret
