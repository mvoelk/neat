
class Species:
    # member is of type Genome
    def __init__(self, first_member, species_id):
        self.leader = first_member
        self.leader_old = None
        self.representative = first_member
        self.members = []
        self.id = species_id
        self.generations_not_improved = 0
        self.age = 0
        self.spawns_required = 0 # number_to_spawn, spawn_amount
        self.max_fitness = 0.0
        self.average_fitness = 0.0

        self.young_age_threshhold = 10
        self.young_age_fitness_bonus = 1.3
        self.old_age_threshold = 50
        self.old_age_fitness_penalty = 0.7

        self.survival_rate = 0.2

        self.add_member(first_member)


    def add_member(self, member): # TODO leader vs max_fitness ???
        member.species_id = self.id
        #print('ADD MEMBER %s TO %s' %(member.id, self.id))
        #print('member_fitness %s, max_fitness %s' %(member.fitness,self.max_fitness))
        #if member.fitness > self.max_fitness:
        #    self.leader = member
        #    self.generations_not_improved = 0
        self.members.append(member)


    def __str__(self):
        s = 'species: id %s, age %s' % (self.id, self.age)
        return s
