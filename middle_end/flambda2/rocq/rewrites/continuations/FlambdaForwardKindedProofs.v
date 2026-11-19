(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaForwardKindedSpec.v. Nothing here
   needs review.

   The relation between a term and its result is that of
   FlambdaForwardProofs.v, taken strict, with the kinds that the pass
   keeps and what it knows of the variables in scope: an application
   that goes on has arguments that are known to have the kinds of the
   parameters and to be of what the handler states of them, and its
   entry is for a handler that gives as many kinds as it has
   parameters. The first run then takes its step where the
   application is: the arguments have values of those kinds
   (FlambdaNormaliseProofs.kinds_hold, [have_kinds_sound]) that are
   of what is stated (FlambdaStatedScopeProofs.are_stated_sound,
   [are_stated_args]), the two tests of the machine pass and the
   parameters are bound. So the first run is undefined only where the
   second is, the rounds of FlambdaForwardProofs.v are read both ways
   ([fwd_round], [fwd_round_back]), and each term refines the other.
   That holds under any variables that have the kinds the pass starts
   from and are of what it starts from knowing, under a static
   environment in which what it starts from knowing of the symbols
   holds ([forward_kinded_under_equivalent_held]). The relation
   carries what is known of the symbols
   (FlambdaStatedSymbolsSpec.symbols_hold): the step of a
   [Let_static] adds what its bindings make known
   (FlambdaStatedSymbolsProofs.statics_stated_bound), and an argument
   that is a symbol is of what is stated by
   FlambdaStatedSymbolsProofs.are_stated_knowing_sound.

   In the body of code the pass starts from the kinds of what a call
   binds, from what the code states of it and from what is known of
   the symbols where the code is bound. Where the body is entered the
   variables are so (FlambdaCodeLetsProofs.code_in_scope_hold,
   FlambdaStatedScopeProofs.code_stated_bound): the machine enters
   code by a closure, with values that it has tested. And the static
   environment of the entry extends the one in which the code was
   bound, so what was known of the symbols holds in it. So the new
   body refines the old one where it is entered under such a static
   environment, and the old one the new ([code_body_forwarded_refines],
   with FlambdaStatedContextProofs.code_refines_stated_where), and a
   group of static bindings and the group with the bodies of its code
   rewritten give static environments under which a term runs the
   same (FlambdaStatedContextProofs.
   related_bindings_equivalent_stated_where). A handler is compared
   where it is entered, in the same way
   (let_cont_handler_refines_stated_where). The induction
   ([forward_kinded_in_code_equivalent_knowing]) is that of
   FlambdaNormaliseProofs.inline_known_calls_equivalent_knowing. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaRoundsProofs FlambdaUnusedParamsSpec FlambdaUnusedParamsProofs
  FlambdaParamAliasesSpec FlambdaParamAliasesProofs FlambdaLoopLetsSpec
  FlambdaLoopLetsProofs FlambdaCodeLetsSpec FlambdaCodeLetsProofs
  FlambdaForwardSpec FlambdaForwardKindedSpec FlambdaForwardProofs.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaValueSpec
  FlambdaSubkindSpec FlambdaStatedScopeSpec FlambdaStatedScopeProofs
  FlambdaStatedSymbolsSpec FlambdaStatedSymbolsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaStatedContextProofs.
Import ListNotations.

Fixpoint forward_kinded_binds_statics (e : expr) :
  ∀ (Y : symbols_stated) (G : kinds_in_scope) (T : stated_in_scope)
    (K : cont_kinds) (F : forwards),
    binds_statics (forward_kinded_under Y G T K F e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros Y G T K F; simpl; try reflexivity.
  - apply forward_kinded_binds_statics.
  - rewrite forward_kinded_binds_statics. reflexivity.
  - rewrite !forward_kinded_binds_statics. reflexivity.
  - rewrite forward_kinded_binds_statics. f_equal.
    generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                  handlers
                ++ F).
    intros F'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_kinded_binds_statics, IH. reflexivity.
Qed.

(* Arguments known to have some kinds have values of those kinds. *)
Lemma have_kinds_sound (G : kinds_in_scope) (args : list simple) :
  ∀ kinds : list flambda_kind,
    have_kinds G args kinds = true -> args_have_kinds G args kinds.
Proof.
  induction args as [ | s args IH]; intros [ | kind kinds] Hhave;
    try discriminate Hhave; intros vars Hhold.
  - exists []. split; reflexivity.
  - simpl in Hhave.
    destruct (kind_of_simple_in_scope G s) as [kind' | ] eqn:Hs;
      [ | discriminate Hhave ].
    apply andb_true_iff in Hhave as [Hkind Hrest].
    apply kind_eqb_eq in Hkind. subst kind'.
    destruct (IH kinds Hrest vars Hhold) as (vs & Hvs & Hkinds).
    destruct (kind_of_simple_in_scope_sound G vars s kind Hhold Hs)
      as (v & Hv & Hkind).
    exists (v :: vs). simpl. rewrite Hv, Hvs, Hkind, Hkinds.
    split; reflexivity.
Qed.

(* Arguments known to be of what is stated have values that are. *)
Lemma are_stated_args (Y : symbols_stated) (T : stated_in_scope)
    (args : list simple) (sks : list param_subkind) :
  are_stated_knowing Y T args sks = true -> args_are_stated Y T args sks.
Proof.
  intros Hstated statics vars vs Hsymbols Hhold Hvs.
  exact
    (are_stated_knowing_sound statics Y T vars args sks vs Hsymbols Hhold
       Hstated Hvs).
Qed.

(* The entry that the pass makes is a strict one for the handler, which
   the body leaves alone. *)
Lemma kinded_entry_fits (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler' body : expr) :
  entry_fits true k params metadata handler'
    (kinded_entry k params metadata handler' body)
  ∧ (∀ x : variable,
      frozen_var (kinded_entry k params metadata handler' body) x ->
      binds_var x body = false)
  ∧ (∀ k0 : continuation,
      frozen_cont (kinded_entry k params metadata handler' body) k0 ->
      binds_cont k0 body = false).
Proof.
  unfold kinded_entry.
  destruct (Nat.eqb (length params) (length (hm_params_kinds metadata)))
    eqn:Hlength.
  - apply forwarding_fits. intros _. apply Nat.eqb_eq, Hlength.
  - split; [ exact I | ].
    split; [ intros x (params0 & target & Hsome & _)
           | intros k0 (params0 & target & Hsome & _) ];
      discriminate Hsome.
Qed.

(* An application that the pass forwards has arguments of the kinds of
   the parameters, which are of what the handler states of them. *)
Lemma goes_forward_kinded (Y : symbols_stated) (G : kinds_in_scope)
    (T : stated_in_scope) (K : cont_kinds) (F : forwards) (ac : apply_cont) :
  goes true Y G T K F ac (forward_kinded_apply_cont Y G T K F ac).
Proof.
  unfold forward_kinded_apply_cont.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hk;
    [ | apply Goes_same ].
  destruct (lookup (ac_continuation ac) K) as [[kinds sks] | ] eqn:Hkinds;
    [ | apply Goes_same ].
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action target
     && have_kinds G (ac_args ac) kinds
     && are_stated_knowing Y T (ac_args ac) sks)
    eqn:Hcheck; [ | apply Goes_same ].
  apply andb_true_iff in Hcheck as [Hcheck Hstated].
  apply andb_true_iff in Hcheck as [Hcheck Hhave].
  apply andb_true_iff in Hcheck as [Hlength _].
  apply Nat.eqb_eq in Hlength.
  apply Goes_keeping; [ exact Hk | exact Hlength | ].
  intros _. exists kinds, sks.
  split; [ exact Hkinds | ].
  split;
    [ exact (have_kinds_sound G _ kinds Hhave)
    | exact (are_stated_args Y T _ sks Hstated) ].
Qed.

(* The same of an arm. Its case more is an application of no arguments
   to a continuation of no parameters whose handler gives no kinds:
   there is nothing to know of kinds, and it goes on with the trap
   action of the handler. *)
Lemma goes_forward_kinded_arm (Y : symbols_stated) (G : kinds_in_scope)
    (T : stated_in_scope) (K : cont_kinds) (F : forwards) (ac : apply_cont) :
  goes true Y G T K F ac (forward_kinded_arm Y G T K F ac).
Proof.
  unfold forward_kinded_arm.
  destruct (forward_of F (ac_continuation ac))
    as [[[ | p params] target] | ] eqn:Hk;
    try apply goes_forward_kinded.
  destruct (lookup (ac_continuation ac) K) as [[[ | kind kinds] sks] | ]
    eqn:Hkinds;
    try apply goes_forward_kinded.
  destruct (ac_args ac) as [ | arg args] eqn:Hargs;
    try apply goes_forward_kinded.
  destruct (no_trap_action ac); [ | apply goes_forward_kinded ].
  replace (forwarded [] [] target) with (forwarded [] (ac_args ac) target)
    by (rewrite Hargs; reflexivity).
  apply Goes_on; [ exact Hk | rewrite Hargs; reflexivity | ].
  intros _. exists [], sks. split; [ exact Hkinds | ].
  rewrite Hargs. split; [ exact (have_kinds_sound G [] [] eq_refl) | ].
  intros statics vars vs _ _ Hvs. injection Hvs as <-.
  destruct sks; reflexivity.
Qed.

Fixpoint forward_kinded_fwd (e : expr) :
  ∀ (Y : symbols_stated) (G : kinds_in_scope) (T : stated_in_scope)
    (K : cont_kinds) (F : forwards),
    safe F e -> fwd true Y G T K F e (forward_kinded_under Y G T K F e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros Y G T K F Hsafe; simpl.
  - destruct (safe_let F x kind d body Hsafe) as [Hbody Hx].
    apply Fwd_let; [ exact Hx | apply forward_kinded_fwd, Hbody ].
  - apply Fwd_let_static, forward_kinded_fwd. exact Hsafe.
  - destruct (safe_let_cont F k params metadata handler body Hsafe)
      as (Hhandler & Hbody & Hparams & Hk).
    set (handler' :=
      forward_kinded_under Y
        (params_in_scope params (hm_params_kinds metadata) G)
        (params_stated params (hm_params_subkinds metadata) T) K F handler).
    destruct (kinded_entry_fits k params metadata handler' body)
      as (Hfits & Hvars & Hconts).
    apply
      (Fwd_let_cont true Y G T K F k params metadata metadata handler handler'
         (kinded_entry k params metadata handler' body));
      [ apply forward_kinded_fwd, Hhandler | reflexivity
      | exact Hparams | exact Hk | exact Hfits | ].
    apply forward_kinded_fwd, safe_cons; assumption.
  - destruct (safe_let_cont_recursive F handlers body Hsafe)
      as [Hbody Hhandlers].
    change (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
              handlers)
      with (plain_forwards handlers).
    assert (Hall :
      Forall
        (fun kh =>
           safe (plain_forwards handlers ++ F) (handler_expr (snd kh))
           ∧ (∀ p : variable, p ∈ handler_params (snd kh) -> var_unfrozen F p)
           ∧ cont_unfrozen F (fst kh))
        handlers).
    { apply Forall_forall. intros [k [params metadata handler]] Hin.
      destruct (Hhandlers k params metadata handler Hin)
        as (Hhandler & Hparams & Hk).
      split; [ apply safe_plain, Hhandler | split; assumption ]. }
    apply Fwd_let_cont_recursive.
    + clear Hsafe Hhandlers. revert Hall.
      generalize (plain_forwards handlers ++ F). intros F' Hall.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        [ apply Forall2_nil | ].
      inversion Hall as [ | kh rest (Hhandler & Hparams & Hk) Hrest].
      subst kh rest.
      apply Forall2_cons; [ | exact (IH Hrest) ].
      simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ].
      split; [ apply forward_kinded_fwd, Hhandler | ].
      split; assumption.
    + clear. unfold handlers_bind_statics.
      generalize (plain_forwards handlers ++ F). intros F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ reflexivity | ].
      rewrite forward_kinded_binds_statics, IH. reflexivity.
    + apply forward_kinded_fwd, safe_plain, Hbody.
  - apply Fwd_apply, Returns_same.
  - apply Fwd_apply_cont, goes_forward_kinded.
  - apply Fwd_switch. clear Hsafe.
    induction arms as [ | arm arms IH]; simpl; [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    split; [ reflexivity | apply goes_forward_kinded_arm ].
  - apply Fwd_invalid.
Qed.

(* The pass that starts from some kinds and from what is known of some
   variables is an equivalence under the variables that have those
   kinds and are of what is known, under the static environment of
   the state. *)
Lemma forward_kinded_under_equivalent_held (Y : symbols_stated)
    (G : kinds_in_scope) (T : stated_in_scope) (e : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  kinds_hold G vars ->
  stated_hold statics T vars ->
  symbols_hold statics Y ->
  equivalent (State e vars conts statics hp rd)
    (State (forward_kinded_under Y G T [] [] e) vars conts statics hp rd).
Proof.
  intros Hhold Hstated Hsymbols.
  assert (Hrel :
    state_fwd true (State e vars conts statics hp rd)
      (State (forward_kinded_under Y G T [] [] e) vars conts statics hp rd)).
  { apply (State_fwd true Y G T [] []);
      [ apply forward_kinded_fwd, safe_none; reflexivity
      | intros k; apply Same_fwd | apply held_nil | exact Hhold
      | exact Hstated | exact Hsymbols ]. }
  apply refines_antisymmetric.
  - exact (rounds_refines (state_fwd true) (fwd_round true) _ _ Hrel).
  - exact
      (rounds_refines (fun t' t => state_fwd true t t') fwd_round_back _ _
         Hrel).
Qed.

Theorem forward_kinded_equivalent : forward_kinded_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  exact
    (forward_kinded_under_equivalent_held [] [] [] e vars conts statics hp rd
       (kinds_hold_nil vars) (stated_hold_nil statics vars)
       (symbols_hold_nil statics)).
Qed.

(* ======================================================================== *)
(* The pass in the bodies of code                                           *)
(* ======================================================================== *)

Fixpoint forward_kinded_in_code_binds_statics (e : expr) :
  ∀ Y : symbols_stated,
    binds_statics (forward_kinded_in_code_under Y e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros Y; simpl; try reflexivity.
  - apply forward_kinded_in_code_binds_statics.
  - rewrite forward_kinded_in_code_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !forward_kinded_in_code_binds_statics. reflexivity.
  - rewrite forward_kinded_in_code_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_kinded_in_code_binds_statics, IH. reflexivity.
Qed.

(* A static environment in which what is known of the symbols holds:
   the property that the laws of FlambdaStatedContextProofs.v are taken
   at. An extension keeps it. *)
Definition symbols_hold_in (Y : symbols_stated) (statics : static_env)
    : Prop :=
  symbols_hold statics Y.

Lemma symbols_hold_in_extend (Y : symbols_stated)
    (statics statics' : static_env) :
  symbols_hold_in Y statics ->
  statics_extend statics statics' ->
  symbols_hold_in Y statics'.
Proof.
  intros Hhold Hextend.
  exact (symbols_hold_extend statics statics' Y Hextend Hhold).
Qed.

(* The body of code and the body as rewritten refine each other where
   the code is entered: by a closure, with values of the kinds of its
   arity that are of what it states of its parameters, under a static
   environment in which what was known of the symbols where the code
   was bound holds. *)
Lemma code_body_forwarded_refines (Y : symbols_stated)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (body : expr) :
  code_refines_stated_where (symbols_hold_in Y) params my_closure metadata
    body
    (forward_kinded_under Y (code_in_scope params my_closure metadata)
       (code_stated params my_closure metadata) [] [] body)
  ∧ code_refines_stated_where (symbols_hold_in Y) params my_closure metadata
      (forward_kinded_under Y (code_in_scope params my_closure metadata)
         (code_stated params my_closure metadata) [] [] body)
      body.
Proof.
  assert (Hboth :
    ∀ (vs : list value) (callee : value) (callee_vars : env)
      (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
      symbols_hold_in Y statics ->
      map kind_of_value vs = unarize (cm_params_arity metadata) ->
      closure_of statics callee <> None ->
      values_of_subkinds statics vs (cm_params_subkinds metadata) = true ->
      bind_params params vs [(my_closure, callee)] = Some callee_vars ->
      equivalent (State body callee_vars conts statics hp rd)
        (State
           (forward_kinded_under Y (code_in_scope params my_closure metadata)
              (code_stated params my_closure metadata) [] [] body)
           callee_vars conts statics hp rd)).
  { intros vs callee callee_vars conts statics hp rd Hsymbols Hkinds Hclosure
      Hstated Hbind.
    apply forward_kinded_under_equivalent_held.
    - destruct (closure_of statics callee) as [[id slots] | ] eqn:Hof;
        [ | contradiction ].
      exact
        (code_in_scope_hold params my_closure metadata vs callee callee_vars
           Hkinds (closure_of_kind _ _ _ _ Hof) Hbind).
    - exact
        (code_stated_bound statics params my_closure metadata vs callee
           callee_vars Hclosure Hstated Hbind).
    - exact Hsymbols. }
  split; intros vs callee callee_vars conts statics hp rd Hsymbols Hkinds
    Hclosure Hstated Hbind;
    [ apply equivalent_refines | apply equivalent_refines, equivalent_sym ];
    exact
      (Hboth vs callee callee_vars conts statics hp rd_ Hsymbols Hkinds
         Hclosure Hstated Hbind).
Qed.

(* A group of static bindings and the group with the bodies of its code
   rewritten have code whose bodies refine each other where they are
   entered. *)
Lemma bindings_forwarded_refine (Y : symbols_stated)
    (bindings : list static_binding) :
  bindings_refine_stated (symbols_hold_in Y) bindings
    (map (forward_kinded_in_binding Y) bindings)
  ∧ bindings_refine_stated (symbols_hold_in Y)
      (map (forward_kinded_in_binding Y) bindings) bindings.
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings IH];
    simpl.
  - repeat split; apply Forall2_nil.
  - destruct IH as [[Hf Hb] [Hf' Hb']].
    destruct
      (code_body_forwarded_refines Y params my_closure metadata code_body)
      as [Hthere Hback].
    repeat split; (apply Forall2_cons; [ | assumption ]);
      apply Binding_code_where, Codes_related_where;
      first
        [ apply forward_kinded_binds_statics
        | symmetry; apply forward_kinded_binds_statics
        | exact Hthere | exact Hback ].
  - destruct IH as [[Hf Hb] [Hf' Hb']].
    repeat split;
      (apply Forall2_cons; [ apply Binding_same_where | assumption ]).
  - destruct IH as [[Hf Hb] [Hf' Hb']].
    repeat split;
      (apply Forall2_cons; [ apply Binding_same_where | assumption ]).
Qed.

(* The pass that starts from what is known of some symbols is an
   equivalence under the static environments in which that holds. *)
Fixpoint forward_kinded_in_code_equivalent_knowing (e : expr) :
  ∀ (Y : symbols_stated) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    symbols_hold statics Y ->
    equivalent (State e vars conts statics hp rd)
      (State (forward_kinded_in_code_under Y e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros Y vars conts statics hp rd Hsymbols; simpl;
    try apply equivalent_refl.
  - apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply forward_kinded_in_code_equivalent_knowing, Hsymbols.
  - (* the body under the static environment of the group as it is,
       then the code of the group *)
    destruct (bindings_forwarded_refine Y bindings) as [Hforwards Hbackwards].
    pose proof
      (related_bindings_equivalent_stated_where (symbols_hold_in Y)
         (symbols_hold_in_extend Y) vars bindings
         (map (forward_kinded_in_binding Y) bindings) statics Hforwards
         Hbackwards)
      as Hrelated.
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings,
      (eval_static_bindings vars
         (map (forward_kinded_in_binding Y) bindings) statics)
        as [extended' | ] eqn:Hbindings';
      try contradiction; [ | exact I ].
    pose proof
      (symbols_hold_extend statics extended Y
         (eval_static_bindings_extend vars _ _ _ Hbindings) Hsymbols)
      as Hext.
    pose proof
      (symbols_hold_extend statics extended' Y
         (eval_static_bindings_extend vars _ _ _ Hbindings') Hsymbols)
      as Hext'.
    apply (equivalent_trans _
      (State (forward_kinded_in_code_under (statics_stated bindings Y) body)
         vars conts extended hp rd)).
    + apply forward_kinded_in_code_equivalent_knowing.
      exact (statics_stated_bound vars extended bindings statics Y Hbindings
        Hext).
    + exact (Hrelated Hext Hext' _ vars conts hp rd).
  - apply (equivalent_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (forward_kinded_in_code_under Y handler))
            body)
         vars conts statics hp rd)).
    + (* the handler is compared where it is entered, under a static
         environment that extends the one of the [Let_cont] *)
      apply refines_antisymmetric;
        apply
          (let_cont_handler_refines_stated_where (symbols_hold_in Y)
             (symbols_hold_in_extend Y));
        try exact Hsymbols;
        intros vs handler_vars conts' statics' hp' rd' Hsymbols' _ _ _;
        apply equivalent_refines; [ | apply equivalent_sym ];
        apply forward_kinded_in_code_equivalent_knowing; exact Hsymbols'.
    + apply let_cont_non_recursive_equivalent_step.
      apply forward_kinded_in_code_equivalent_knowing, Hsymbols.
  - apply (equivalent_trans _
      (State
         (Let_cont_recursive
            (map (fun '(k, h) => (k, forward_kinded_in_code_handler Y h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        let handlers' :=
          map (fun '(k, h) => (k, forward_kinded_in_code_handler Y h))
            handlers
        in
        handlers_refine_stated (symbols_hold_in Y) vars handlers handlers'
        ∧ handlers_refine_stated (symbols_hold_in Y) vars handlers'
            handlers).
      { induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - repeat split; apply Forall2_nil.
        - destruct IH as [[Hforwards Hbackwards] [Hforwards' Hbackwards']].
          repeat split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hsymbols' _ _ _;
            try (apply equivalent_refines,
                   forward_kinded_in_code_equivalent_knowing; assumption);
            apply equivalent_refines, equivalent_sym,
              forward_kinded_in_code_equivalent_knowing; assumption. }
      destruct Hhandlers as [Hforwards Hbackwards].
      assert (Hbinds :
        handlers_bind_statics
          (map (fun '(k, h) => (k, forward_kinded_in_code_handler Y h))
             handlers)
        = handlers_bind_statics handlers).
      { unfold handlers_bind_statics. clear.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite forward_kinded_in_code_binds_statics, IH. reflexivity. }
      apply refines_antisymmetric;
        apply
          (let_cont_recursive_handlers_refine_stated_where (symbols_hold_in Y)
             (symbols_hold_in_extend Y));
        try assumption; congruence.
    + apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply forward_kinded_in_code_equivalent_knowing, Hsymbols.
Qed.

Theorem forward_kinded_in_code_equivalent :
  forward_kinded_in_code_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  exact
    (forward_kinded_in_code_equivalent_knowing e [] vars conts statics hp rd_
       (symbols_hold_nil statics)).
Qed.
