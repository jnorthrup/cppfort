
/Users/jim/work/cppfort/micro-tests/results/memory/mem110-cache-line-alignment/mem110-cache-line-alignment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_cache_linev>:
100000360: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000364: 910003fd    	mov	x29, sp
100000368: d101c3e9    	sub	x9, sp, #0x70
10000036c: 927ae53f    	and	sp, x9, #0xffffffffffffffc0
100000370: 52800548    	mov	w8, #0x2a               ; =42
100000374: b90003e8    	str	w8, [sp]
100000378: b94003e0    	ldr	w0, [sp]
10000037c: 910003bf    	mov	sp, x29
100000380: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: d10083ff    	sub	sp, sp, #0x20
10000038c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000390: 910043fd    	add	x29, sp, #0x10
100000394: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000398: 97fffff2    	bl	0x100000360 <__Z15test_cache_linev>
10000039c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a0: 910083ff    	add	sp, sp, #0x20
1000003a4: d65f03c0    	ret
