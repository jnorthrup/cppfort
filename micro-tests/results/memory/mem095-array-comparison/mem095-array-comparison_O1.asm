
/Users/jim/work/cppfort/micro-tests/results/memory/mem095-array-comparison/mem095-array-comparison_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z21test_array_comparisonv>:
1000003b0: d280000a    	mov	x10, #0x0               ; =0
1000003b4: 52800028    	mov	w8, #0x1                ; =1
1000003b8: aa0a03e9    	mov	x9, x10
1000003bc: f100095f    	cmp	x10, #0x2
1000003c0: 54000060    	b.eq	0x1000003cc <__Z21test_array_comparisonv+0x1c>
1000003c4: 9100052a    	add	x10, x9, #0x1
1000003c8: 35ffff88    	cbnz	w8, 0x1000003b8 <__Z21test_array_comparisonv+0x8>
1000003cc: f100053f    	cmp	x9, #0x1
1000003d0: 1a9f97e0    	cset	w0, hi
1000003d4: d65f03c0    	ret

00000001000003d8 <_main>:
1000003d8: d2800009    	mov	x9, #0x0                ; =0
1000003dc: 52800048    	mov	w8, #0x2                ; =2
1000003e0: 5280002a    	mov	w10, #0x1               ; =1
1000003e4: f100093f    	cmp	x9, #0x2
1000003e8: 54000080    	b.eq	0x1000003f8 <_main+0x20>
1000003ec: 91000529    	add	x9, x9, #0x1
1000003f0: 35ffffaa    	cbnz	w10, 0x1000003e4 <_main+0xc>
1000003f4: d1000528    	sub	x8, x9, #0x1
1000003f8: f100051f    	cmp	x8, #0x1
1000003fc: 1a9f97e0    	cset	w0, hi
100000400: d65f03c0    	ret
