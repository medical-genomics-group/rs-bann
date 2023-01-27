require(BEDMatrix)
SNPs1 <- BEDMatrix("ukb_imp_v3_bm_1", simple_names=TRUE)
SNPs2 <- BEDMatrix("ukb_imp_v3_bm_2", simple_names=TRUE)
lds <- read.table("bm_grouped_bim_file_maf_score", header=T, stringsAsFactors=FALSE)
spacing <- seq(from=1, to=nrow(lds), by=59)
qtl_blocks <- as.character(lds$SNP)
h2 = 0.6

for(i in 1:20)
{
    snp_ids <- c()
    
    for(k in 1:10000)
    {
        snp_ids[k] <- as.character(sample(qtl_blocks[(spacing[k+1]+1):spacing[k+2]],1))
    }
    
    markers1 <- SNPs1[,snp_ids]
    markers2 <- SNPs2[,snp_ids]
    markers <- rbind(markers1,markers2)
    
    for(m in seq_len(ncol(markers)))
    {
        markers[,m][is.na(markers[,m])] <- mean(markers[,m], na.rm=TRUE)
    }
    
    M = ncol(markers)
    N = nrow(markers)
    ldld <- lds[which(lds$SNP %in% snp_ids),]
    p_var <- 2* ldld$MAF * (1 - ldld$MAF)
    power1 <- (0.5)
    power2 <- (-0.5)
    b_adj <- (ldld$ldscore^power1) * (p_var^power2)
    var_b <- h2/M
    markers <- scale(markers)
    beta <- c()
    
    for(j in 1:M)
    {
        beta[j] <- rnorm(1,0,sqrt(var_b)*(b_adj[j]))
    }
    
    beta <- (scale(beta)*sqrt(var_b))
    g <- markers %*% beta
    e <- rnorm(N,0,sqrt(1-var(g)))
    y = g + e

    # output phenotype
    #system(paste("mkdir -p /work/ext-unil-ctgg/robinson/benchmarking/random/sim_",i,sep=''))
    phen_out <- paste("/work/ext-unil-ctgg/robinson/benchmarking/random/sim_",i,"/sim_05m05.phen",sep='')
    phen <- data.frame("fid" = as.character(rownames(markers)), "iid" = as.character(rownames(markers)), "phen" = y)
    write.table(phen, phen_out, row.names=FALSE, col.names=FALSE, quote=FALSE)

    # output genotype
    gen_out <- paste("/work/ext-unil-ctgg/robinson/benchmarking/random/sim_",i,"/sim_05m05.trueG",sep='')
    gen <- data.frame("fid" = as.character(rownames(markers)), "iid" = as.character(rownames(markers)), "gen" = g)
    write.table(gen, gen_out, row.names=FALSE, col.names=FALSE, quote=FALSE)

    # output beta values
    beta_out <- paste("/work/ext-unil-ctgg/robinson/benchmarking/random/sim_",i,"/sim_05m05.trueB",sep='')
    b_out <- data.frame("SNP" = as.character(colnames(markers)), "beta" = beta)
    write.table(b_out, beta_out, row.names=FALSE, col.names=FALSE, quote=FALSE)

    #output h2
    h2_out <- paste("/work/ext-unil-ctgg/robinson/benchmarking/random/sim_",i,"/sim_05m05.trueh2",sep='')
    h2_est <- var(g) / var(y)
    write.table(h2_est, h2_out, row.names=FALSE, col.names=FALSE, quote=FALSE)
}
