
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf095-mutual-recursion/cf095-mutual-recursion_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z7is_eveni>:
100000360: 340000a0    	cbz	w0, 0x100000374 <__Z7is_eveni+0x14>
100000364: 7100041f    	cmp	w0, #0x1
100000368: 540000a0    	b.eq	0x10000037c <__Z7is_eveni+0x1c>
10000036c: 51000800    	sub	w0, w0, #0x2
100000370: 35ffffa0    	cbnz	w0, 0x100000364 <__Z7is_eveni+0x4>
100000374: 52800020    	mov	w0, #0x1                ; =1
100000378: d65f03c0    	ret
10000037c: 52800000    	mov	w0, #0x0                ; =0
100000380: d65f03c0    	ret

0000000100000384 <__Z6is_oddi>:
100000384: 34000100    	cbz	w0, 0x1000003a4 <__Z6is_oddi+0x20>
100000388: 51000408    	sub	w8, w0, #0x1
10000038c: 340000a8    	cbz	w8, 0x1000003a0 <__Z6is_oddi+0x1c>
100000390: 7100051f    	cmp	w8, #0x1
100000394: 540000a0    	b.eq	0x1000003a8 <__Z6is_oddi+0x24>
100000398: 51000908    	sub	w8, w8, #0x2
10000039c: 35ffffa8    	cbnz	w8, 0x100000390 <__Z6is_oddi+0xc>
1000003a0: 52800020    	mov	w0, #0x1                ; =1
1000003a4: d65f03c0    	ret
1000003a8: 52800000    	mov	w0, #0x0                ; =0
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: 52800020    	mov	w0, #0x1                ; =1
1000003b4: d65f03c0    	ret
