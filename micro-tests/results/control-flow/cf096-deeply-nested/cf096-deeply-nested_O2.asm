
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf096-deeply-nested/cf096-deeply-nested_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_deeply_nestedi>:
100000360: 52800028    	mov	w8, #0x1                ; =1
100000364: 52800049    	mov	w9, #0x2                ; =2
100000368: 7100781f    	cmp	w0, #0x1e
10000036c: 5280006a    	mov	w10, #0x3               ; =3
100000370: 1a8a954a    	cinc	w10, w10, hi
100000374: 7100541f    	cmp	w0, #0x15
100000378: 1a8a3129    	csel	w9, w9, w10, lo
10000037c: 71002c1f    	cmp	w0, #0xb
100000380: 1a893108    	csel	w8, w8, w9, lo
100000384: 7100041f    	cmp	w0, #0x1
100000388: 1a88b3e0    	csel	w0, wzr, w8, lt
10000038c: d65f03c0    	ret

0000000100000390 <_main>:
100000390: 52800060    	mov	w0, #0x3                ; =3
100000394: d65f03c0    	ret
