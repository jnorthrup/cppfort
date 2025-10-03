
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar054-bit-count/ar054-bit-count_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_bit_countj>:
100000360: 34000120    	cbz	w0, 0x100000384 <__Z14test_bit_countj+0x24>
100000364: aa0003e8    	mov	x8, x0
100000368: 52800000    	mov	w0, #0x0                ; =0
10000036c: 12000109    	and	w9, w8, #0x1
100000370: 0b090000    	add	w0, w0, w9
100000374: 53017d09    	lsr	w9, w8, #1
100000378: 7100051f    	cmp	w8, #0x1
10000037c: aa0903e8    	mov	x8, x9
100000380: 54ffff68    	b.hi	0x10000036c <__Z14test_bit_countj+0xc>
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: 52800080    	mov	w0, #0x4                ; =4
10000038c: d65f03c0    	ret
